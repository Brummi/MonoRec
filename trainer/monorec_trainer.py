import operator

import numpy as np
import torch
from torch.nn import DataParallel
from torchvision.utils import make_grid

from trainer.trainer import Trainer
from utils import map_fn, operator_on_dict


class MonoRecTrainer(Trainer):
    def __init__(self, model, loss, metrics, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None, options=()):
        super().__init__(model, loss, metrics, optimizer, config, data_loader, valid_data_loader, lr_scheduler, options)
        self.compute_mono_pred = config["trainer"].get("compute_mono_pred", True)
        self.compute_stereo_pred = config["trainer"].get("compute_stereo_pred", True)
        self.compute_mask = config["trainer"].get("compute_mask", True)
        self.mult_mask_on_cv = config["trainer"].get("mult_mask_on_cv", False)
        self.concat_mono_stereo = config["trainer"].get("concat_mono_stereo", False)

    def _feed(self, data_dict):
        if isinstance(self.model, DataParallel):
            model = self.model.module
        else:
            model = self.model

        data_dict["inv_depth_min"] = data_dict["keyframe"].new_tensor([model.inv_depth_min_max[0]])
        data_dict["inv_depth_max"] = data_dict["keyframe"].new_tensor([model.inv_depth_min_max[1]])
        data_dict["cv_depth_steps"] = data_dict["keyframe"].new_tensor([model.cv_depth_steps], dtype=torch.int32)

        orig_data_dict = dict(data_dict)

        if model.augmenter is not None and model.training: model.augmenter(data_dict)

        # Get image features
        data_dict["image_features"] = model._feature_extractor(data_dict["keyframe"] + .5)

        model.use_mono = False
        model.use_stereo = True
        model.cv_module.use_mono = False
        model.cv_module.use_stereo = True

        if self.compute_stereo_pred:
            # Compute stereo CV
            with torch.no_grad():
                orig_data_dict = model.cv_module(orig_data_dict)

            if model.augmenter is not None and model.training:
                data_dict["cost_volume"] = model.augmenter.single_apply(orig_data_dict["cost_volume"])
                data_dict["single_frame_cvs"] = [model.augmenter.single_apply(sfcv) for sfcv in orig_data_dict["single_frame_cvs"]]
            else:
                data_dict["cost_volume"] = orig_data_dict["cost_volume"]
                data_dict["single_frame_cvs"] = orig_data_dict["single_frame_cvs"]

            # Compute stereo depth
            if not self.concat_mono_stereo:
                with torch.no_grad():
                    data_dict = model.depth_module(data_dict)
            else:
                data_dict = model.depth_module(data_dict)
            stereo_pred = [(1 - pred) * model.inv_depth_min_max[1] + pred * model.inv_depth_min_max[0] for pred in data_dict["predicted_inverse_depths"]]
        else:
            stereo_pred = None

        model.use_mono = True
        model.use_stereo = False
        model.cv_module.use_mono = True
        model.cv_module.use_stereo = False

        # Compute mono CV
        with torch.no_grad():
            orig_data_dict = model.cv_module(orig_data_dict)
        if model.augmenter is not None and model.training:
            data_dict["cost_volume"] = model.augmenter.single_apply(orig_data_dict["cost_volume"])
            data_dict["single_frame_cvs"] = [model.augmenter.single_apply(sfcv) for sfcv in orig_data_dict["single_frame_cvs"]]
        else:
            data_dict["cost_volume"] = orig_data_dict["cost_volume"]
            data_dict["single_frame_cvs"] = orig_data_dict["single_frame_cvs"]

        # Compute mask
        if self.compute_mask:
            data_dict = model.att_module(data_dict)
            if self.mult_mask_on_cv:
                data_dict["cost_volume"] *= (1 - data_dict["cv_mask"])
        else:
            data_dict["cv_mask"] = data_dict["cost_volume"].new_zeros(data_dict["cost_volume"][:, :1, :, :].shape, requires_grad=False)
        if self.compute_mono_pred:
            # Compute mono depth
            data_dict = model.depth_module(data_dict)
            mono_pred = [(1 - pred) * model.inv_depth_min_max[1] + pred * model.inv_depth_min_max[0] for pred in data_dict["predicted_inverse_depths"]]
        else:
            mono_pred = [data_dict["cost_volume"].new_zeros(data_dict["cost_volume"][:, :1, :, :].shape, requires_grad=False)]

        # Prepare dict
        data_dict["mono_pred"] = mono_pred
        data_dict["stereo_pred"] = stereo_pred
        data_dict["predicted_inverse_depths"] = mono_pred
        data_dict["result"] = mono_pred[0]
        data_dict["mask"] = data_dict["cv_mask"]

        if model.augmenter is not None and model.training: model.augmenter.revert(data_dict)

        if self.concat_mono_stereo:
            data_dict["keyframe"] = torch.cat(2 * [data_dict["keyframe"]], dim=0)
            data_dict["keyframe_pose"] = torch.cat(2 * [data_dict["keyframe_pose"]], dim=0)
            data_dict["keyframe_intrinsics"] = torch.cat(2 * [data_dict["keyframe_intrinsics"]], dim=0)
            data_dict["stereoframe"] = torch.cat(2 * [data_dict["stereoframe"]], dim=0)
            data_dict["stereoframe_pose"] = torch.cat(2 * [data_dict["stereoframe_pose"]], dim=0)
            data_dict["stereoframe_intrinsics"] = torch.cat(2 * [data_dict["stereoframe_intrinsics"]], dim=0)
            data_dict["frames"] = [torch.cat(2 * [frame], dim=0) for frame in data_dict["frames"]]
            data_dict["poses"] = [torch.cat(2 * [pose], dim=0) for pose in data_dict["poses"]]
            data_dict["intrinsics"] = [torch.cat(2 * [intrinsics], dim=0) for intrinsics in data_dict["intrinsics"]]
            data_dict["mask"] = torch.cat(2 * [data_dict["mask"]], dim=0)
            data_dict["cv_mask"] = torch.cat(2 * [data_dict["cv_mask"]], dim=0)
            data_dict["target"] = torch.cat(2 * [data_dict["target"]], dim=0)

            data_dict["predicted_inverse_depths"] = [torch.cat([m, s], dim=0) for m, s in zip(mono_pred, stereo_pred)]
            data_dict["result"] = data_dict["predicted_inverse_depths"][0]

        loss_dict = self.loss(data_dict, alpha=self.alpha, roi=self.roi, options=self.options)

        return loss_dict, data_dict


    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_loss_dict = {}
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = to(data, self.device), to(target, self.device)
            data["target"] = target
            data["optimizer"] = self.optimizer

            self.optimizer.zero_grad()

            loss_dict, data = self._feed(data)

            loss_dict = map_fn(loss_dict, torch.mean)

            loss = loss_dict["loss"]
            if loss.requires_grad:
                loss.backward()
            self.optimizer.step()
            loss_dict = map_fn(loss_dict, torch.detach)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            self.writer.add_scalar('loss', loss.item())
            for loss_component, v in loss_dict.items():
                self.writer.add_scalar(f"loss_{loss_component}", v.item())

            total_loss += loss.item()
            total_loss_dict = operator_on_dict(total_loss_dict, loss_dict, operator.add)
            metrics, valid = self._eval_metrics(data, training=True)
            total_metrics += metrics

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
            'metrics': (total_metrics / self.len_epoch).tolist()
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
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = to(data,  self.device), to(target, self.device)
                data["target"] = target

                loss_dict, data = self._feed(data)
                loss_dict = map_fn(loss_dict, torch.mean)
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
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

        for loss_component, v in total_val_loss_dict.items():
            result[f"val_loss_{loss_component}"] = v.item() / len(self.valid_data_loader)

        return result


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