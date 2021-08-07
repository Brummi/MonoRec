import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils import seed_rng
from utils.parse_config import ConfigParser
from trainer.monorec_trainer import MonoRecTrainer


def main(config, options=()):
    seed_rng(0)
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    if "val_data_loader" in config.config:
        valid_data_loader = config.initialize("val_data_loader", module_data)
    else:
        valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)
    logger.info(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # get function handles of loss and metrics
    if "loss_module" in config.config:
        loss = config.initialize("loss_module", module_loss)
    else:
        loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = MonoRecTrainer(model, loss, metrics, optimizer,
                             config=config,
                             data_loader=data_loader,
                             valid_data_loader=valid_data_loader,
                             lr_scheduler=lr_scheduler,
                             options=options)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--options', default=[], nargs='+')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    print(config.config)
    main(config, config.args.options)
