import argparse
import logging
import ipdb
import os
import sys
import torch
import random
import importlib
import yaml
from box import Box
from pathlib import Path

import src


def main(args):
    # load the config 
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)

    # copy the config
    loggging.info(f'Save the config to "{config.main.save_dir}".')
    with open(saved_dir / 'config.yaml', 'w+') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # fix the random seed
    random.seed(config.main.random_seed)
    torch.manual_seed(random.getstate()[1][1])
    torch.cuda.manual_seed_all(random.getstate()[1][1])
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False

    # device
    logging.info('Create the device.')
    if 'cuda' in config.trainer.kwargs.device and not torch.cuda.is_available():
        raise ValueError("The cuda is not available.")
    device = torch.device(config.trainer.kwargs.device)

    # dataset
    logging.info('Create the training and validation datasets.')
    data_dir = Path(config.dataset.kwargs.data_dir)
    config.dataset.kwargs.update(data_dir=data_dir, type='train')
    train_dataset = _get_instance(src.data.datasets, config.dataset)
    config.dataset.kwargs.update(data_dir=data_dir, type='valid')
    valid_dataset = _get_instance(src.data.datasets, config.dataset)

    # dataloader
    logging.info('Create the training and validation dataloaders.')
    cls = getattr(src.data.datasets, config.dataset.name)
    train_batch_size, valid_batch_size = config.dataloader.kwargs.pp('train_batch_size'), config.dataloader.kwargs.pop('valid_batch_size')
    config.dataloader.kwargs.update(collate_fn=getattr(cls, 'collate_fn', None), batch_size=train_batch_size)
    train_dataloader = _get_instance(src.data.dataloader, config.dataloader, train_dataset)
    config.dataloader.kwargs.update(batch_size=valid_batch_size)
    valid_dataloader = _get_instance(src.data.dataloader, config.dataloader, valid_dataset)

    # net
    logging.infor('Create the network architecture.')
    net = _get_instance(src.model.nets, config.net)

    # loss
    logging.info('Create the loss functions and the corresponding weights')
    loss_fns, loss_weights = [], []
    defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
    for config_loss in config.losses:
        if config_loss.name in defaulted_loss_fns:
            loss_fn = _get_instance(torch.nn, config_loss)
        else:
            loss_fn = _get_instance(src.model.losses, config.loss)
        loss_fns.append(loss_fn)
        loss_weights.append(config_loss.weight)

    # metric
    logging.info('Create the metric functions.')
    metric_fns = [_get_instance(src.model.metrics, config_metric for config_metric in config_metrics)]

    # optimizer
    logging.info('Create the optimizer')
    optimizer = _get_instance(torch.optim, config.optimizer, net.parameters())

    # learning rate scheduler
    logging.info('Create the learning rate scheduler')
    lr_scheduler = _get_instance(torch.optim.lr_scheduler, config.lr_scheduler, optimizer) if config.get('lr_scheduler') else None

    # logger
    logging.info('Create the logger')
    config.logger.kwargs.update(log_dir=saved_dir / 'log', net=net, dummy_input=torch.randn(tuple(config.logger.kwargs.dummy_input)))
    logger = _get_instance(src.callbacks.loggers, config.logger)

    # monitor
    logging.info('Create the monitor')
    config.monitor.kwargs.update(checkpoints_dir=saved_dir / 'checkpoints')
    monitor = _get_instance(src.callbacks.monitor, config.monitor)

    # trainer
    logging.info('Create the trainer.')
    kwargs = {'device': device, 
              'train_dataloader': train_dataloader, 
              'vliad_dataloader': valid_dataloader, 
              'net': net, 
              'loss_fns': loss_fns, 
              'loss_weights': loss_weights, 
              'metric_fns': metric_fns, 
              'optimizer': optimizer, 
              'lr_scheduler': lr_scheduler, 
              'logger': logger, 
              'monitor': monitor}
    config.trainer.kwargs.update(kwargs)
    trainer = _get_instance(src.runner.trainer, config.trainer)

    # load previous checkpoint
    loaded_path = config.main.get('loaded_path')
    if loaded_path:
        logging.info(f'Load the previous checkpoint from "{loaded_path}.')
        trainer.load(Path(loaded_path))
        logging.info('Resume training')
    else:
        logging.info('Start training.')
    trainer.train()
    logging.info('End training')


def _get_instance(module, config, *args):
    """
    Args:
        module (module): the python module
        config (Box): the config to create the class object

    Returns:
        instance (object): the class object defined in the module
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **kwargs) if kwargs else cls(*args)


def _parse_args():
    parser = argparse.ArgumentParser(description="The script for the training and the testing.")
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    parser.add_argument('--test', action='store_true', help='Perform the training if specified; otherwise perform the testing.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)