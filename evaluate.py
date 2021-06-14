import argparse
import json
import pathlib
from pathlib import Path

import numpy as np

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from evaluater import Evaluater
from utils.parse_config import ConfigParser


def main(config: ConfigParser):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build model architecture, then print to console

    if "arch" in config.config:
        models = [config.initialize('arch', module_arch)]
    else:
        models = config.initialize_list("models", module_arch)

    results = []

    for i, model in enumerate(models):
        model_dict = dict(model.__dict__)
        keys = list(model_dict.keys())
        for k in keys:
            if k.startswith("_"):
                model_dict.__delitem__(k)
            elif type(model_dict[k]) == np.ndarray:
                model_dict[k] = list(model_dict[k])


        dataset_dict = dict(data_loader.dataset.__dict__)
        keys = list(dataset_dict.keys())
        for k in keys:
            if k.startswith("_"):
                dataset_dict.__delitem__(k)
            elif type(dataset_dict[k]) == np.ndarray:
                dataset_dict[k] = list(dataset_dict[k])
            elif isinstance(dataset_dict[k], pathlib.PurePath):
                dataset_dict[k] = str(dataset_dict[k])


        logger.info(model_dict)
        logger.info(dataset_dict)
        evaluater = Evaluater(model, loss, metrics, config=config, data_loader=data_loader)
        result = evaluater.eval(i)
        result["metrics"] = result["metrics"]
        del model
        result["metrics_info"] = [metric.__name__ for metric in metrics]
        logger.info(result)
        results.append({
            "model": model_dict,
            "dataset": dataset_dict,
            "result": result
        })

    save_file = Path(config.log_dir) / "results.json"
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Deeptam Evaluation')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    config = ConfigParser(args)
    print(config.config)
    main(config)
