import operator

import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, map_fn, operator_on_dict, LossWrapper, ValueFader

import time

class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, options=[]):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader

        len_epoch = config["trainer"].get("len_epoch", None)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config["trainer"].get("log_step", int(np.sqrt(data_loader.batch_size)))
        self.val_log_step = config["trainer"].get("val_step", 1)
        self.roi = config["trainer"].get("roi")
        self.roi_train = config["trainer"].get("roi_train", self.roi)
        self.alpha = config["trainer"].get("alpha", None)
        self.max_distance = config["trainer"].get("max_distance", None)
        self.val_avg = config["trainer"].get("val_avg", True)
        self.save_multiple = config["trainer"].get("save_multiple", False)
        self.invert_output_images = config["trainer"].get("invert_output_images", True)
        self.wrap_loss_in_module = config["trainer"].get("wrap_loss_in_module", False)
        self.value_faders = config["trainer"].get("value_faders", {})
        self.options = options

        if self.wrap_loss_in_module:
            self.loss = LossWrapper(loss_function=self.loss, roi=self.roi, options=self.options)

        if isinstance(loss, torch.nn.Module) or self.wrap_loss_in_module:
            self.module_loss = True
            self.loss.to(self.device)
            if len(self.device_ids) > 1:
                self.loss.num_devices = len(self.device_ids)
                self.model = torch.nn.DataParallel(torch.nn.Sequential(self.model.module, self.loss), self.device_ids)
            else:
                self.model = torch.nn.Sequential(self.model, self.loss)
        else:
            self.module_loss = False

        self.value_faders = {k: ValueFader(v[0], v[1]) for k, v in self.value_faders.items()}

    def _eval_metrics(self, data_dict, training=False):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(data_dict, self.roi, self.max_distance)
            if (not self.val_avg) or training:
                self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        if np.any(np.isnan(acc_metrics)):
            acc_metrics = np.zeros(len(self.metrics))
            valid = np.zeros(len(self.metrics))
        else:
            valid = np.ones(len(self.metrics))
        return acc_metrics, valid

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_loss_dict = {}
        total_metrics = np.zeros(len(self.metrics))
        total_metrics_valid = np.zeros(len(self.metrics))

        fade_values = {k: torch.tensor([fader.get_value(epoch)]) for k, fader in self.value_faders.items()}

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data.update(fade_values)
            data, target = to(data, self.device), to(target, self.device)
            data["target"] = target
            # data["optimizer"] = self.optimizer

            start_time = time.time()

            self.optimizer.zero_grad()

            if not self.module_loss:
                data = self.model(data)
                loss_dict = self.loss(data, self.alpha, self.roi_train, options=self.options)
            else:
                data, loss_dict = self.model(data)

            loss_dict = map_fn(loss_dict, torch.sum)

            loss = loss_dict["loss"]
            if loss.requires_grad:
                loss.backward()

            self.optimizer.step()

            # print("Forward time: ", (time.time() - start_time))

            loss_dict = map_fn(loss_dict, torch.detach)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            self.writer.add_scalar('loss', loss.item())
            for loss_component, v in loss_dict.items():
                self.writer.add_scalar(f"loss_{loss_component}", v.item())

            total_loss += loss.item()
            total_loss_dict = operator_on_dict(total_loss_dict, loss_dict, operator.add)
            metrics, valid = self._eval_metrics(data, True)
            total_metrics += metrics
            total_metrics_valid += valid

            if self.writer.step % self.log_step == 0:
                img_count = min(data["keyframe"].shape[0], 8)

                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Loss_dict: {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    loss_dict))

                if "mask" in data:
                    if self.invert_output_images:
                        result = torch.clamp(1 / data["result"][:img_count], 0, 100).cpu()
                        result /= torch.max(result) * 2 / 3
                    else:
                        result = data["result"][:img_count].cpu()
                    mask = data["mask"][:img_count].cpu()
                    img = torch.cat([result, mask], dim=2)
                else:
                    if self.invert_output_images:
                        img = torch.clamp(1 / data["result"][:img_count], 0, 100).cpu()
                    else:
                        img = data["result"][:img_count].cpu()

                self.writer.add_image('input', make_grid(to(data["keyframe"][:img_count], "cpu"), nrow=2, normalize=True))
                self.writer.add_image('output', make_grid(img , nrow=2, normalize=True))
                self.writer.add_image('ground_truth', make_grid(to(torch.clamp(infnan_to_zero(1 / data["target"][:img_count]), 0, 100), "cpu"), nrow=2, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / total_metrics_valid).tolist()
        }
        for loss_component, v in total_loss_dict.items():
            log[f"loss_{loss_component}"] = v.item() / self.len_epoch

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()

        total_val_loss = 0
        total_val_loss_dict = {}
        total_val_metrics = np.zeros(len(self.metrics))
        total_val_metrics_valid = np.zeros(len(self.metrics))

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = to(data,  self.device), to(target, self.device)
                data["target"] = target

                if not self.module_loss:
                    data = self.model(data)
                    loss_dict = self.loss(data, self.alpha, self.roi_train, options=self.options)
                else:
                    data, loss_dict = self.model(data)

                loss_dict = map_fn(loss_dict, torch.sum)
                loss = loss_dict["loss"]

                img_count = min(data["keyframe"].shape[0], 8)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                if not self.val_avg:
                    self.writer.add_scalar('loss', loss.item())
                    for loss_component, v in loss_dict.items():
                        self.writer.add_scalar(f"loss_{loss_component}", v.item())
                total_val_loss += loss.item()
                total_val_loss_dict = operator_on_dict(total_val_loss_dict, loss_dict, operator.add)
                metrics, valid = self._eval_metrics(data)
                total_val_metrics += metrics
                total_val_metrics_valid += valid
                if batch_idx % self.val_log_step == 0:
                    if "mask" in data:
                        if self.invert_output_images:
                            result = torch.clamp(1 / data["result"][:img_count], 0, 100).cpu()
                            result /= torch.max(result) * 2 / 3
                        else:
                            result = data["result"][:img_count].cpu()
                        mask = data["mask"][:img_count].cpu()
                        img = torch.cat([result, mask], dim=2)
                    else:
                        if self.invert_output_images:
                            img = torch.clamp(1 / data["result"][:img_count], 0, 100).cpu()
                        else:
                            img = data["result"][:img_count].cpu()

                    self.writer.add_image('input', make_grid(to(data["keyframe"][:img_count], "cpu"), nrow=2, normalize=True))
                    self.writer.add_image('output', make_grid(img, nrow=2, normalize=True))
                    self.writer.add_image('ground_truth', make_grid(to(torch.clamp(infnan_to_zero(1 / data["target"][:img_count]), 0, 100), "cpu"), nrow=2, normalize=True))

        if self.val_avg:
            len_val = len(self.valid_data_loader)
            self.writer.add_scalar('loss', total_val_loss / len_val)
            for i, metric in enumerate(self.metrics):
                self.writer.add_scalar('{}'.format(metric.__name__), total_val_metrics[i] / len_val)
            for loss_component, v in total_val_loss_dict.items():
                self.writer.add_scalar(f"loss_{loss_component}", v.item() / len_val)

        result = {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / total_val_metrics_valid).tolist()
        }

        for loss_component, v in total_val_loss_dict.items():
            result[f"val_loss_{loss_component}"] = v.item() / len(self.valid_data_loader)

        return result

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def to(data, device):
    if isinstance(data, dict):
        return {k: to(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to(v, device) for v in data]
    else:
        return data.to(device)


def infnan_to_zero(t:torch.Tensor()):
    t[torch.isinf(t)] = 0
    t[torch.isnan(t)] = 0
    return t