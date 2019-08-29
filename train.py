import os
import json
import argparse
import torch
import collections
import data_loader.data_loaders as module_data
from data_loader.data_loaders import load_word_dict, load_rel_dict
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import pickle
import numpy as np
import random

import pdb


def get_instance(module, name, config, **kwargs):
    return getattr(module, config[name]["type"])(**kwargs, **config[name]["args"])


def main(config):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    #init random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # setup data_loader instances
    word2id, word2vec = load_word_dict(
        os.path.join(config["data_dir"], "word_vec.json")
    )
    rel2id = load_rel_dict(os.path.join(config["data_dir"], "rel2id.json"))
    train_data_loader = get_instance(
        module_data,
        "data_loader",
        config,
        mode="train",
        src="train",
        shuffle=True,
        word2id=word2id,
        rel2id=rel2id,
        validation_split=0,
    )

    test_data_loader = get_instance(
        module_data,
        "data_loader",
        config,
        mode="val",
        src="val",
        shuffle=False,
        word2id=word2id,
        rel2id=rel2id,
    )

    # build model architecture
    model = get_instance(
        module_arch, "arch", config, word_vec_mat=word2vec, relation_num=len(rel2id)
    )
    print(model)

    weight = torch.tensor(
        1 / ((train_data_loader.rel2count + 1) ** config["label_reweight"])
    ).float()
    print("weight for relations:")
    print(weight)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config["loss"])
    train_metrics = [getattr(module_metric, met) for met in config["train_metrics"]]
    eval_metrics = [getattr(module_metric, met) for met in config["eval_metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, "optimizer", config, params=trainable_params)
    lr_scheduler = get_instance(
        torch.optim.lr_scheduler, "lr_scheduler", config, optimizer=optimizer
    )

    trainer = Trainer(
        model,
        loss,
        train_metrics,
        eval_metrics,
        optimizer,
        config=config,
        data_loader=train_data_loader,
        valid_data_loader=test_data_loader,
        lr_scheduler=lr_scheduler,
        weight=weight,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        required=True,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target=("optimizer", "args", "lr")
        ),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target=("data_loader", "args", "batch_size"),
        ),
        CustomArgs(
            ["--weight", "--label_reweight"], type=float, target=("label_reweight")
        ),
    ]
    config = ConfigParser(args, options)

    main(config)
