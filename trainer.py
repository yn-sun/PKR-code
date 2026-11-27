import copy
import logging
import os
import sys

import numpy as np
import torch

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args["increment"])
    os.makedirs(logs_name, exist_ok=True)
    logfilename = "{}/{}_{}_{}".format(logs_name, args["prefix"], args["seed"], args["convnet_type"])

    for h in logging.root.handlers[:]:  # fresh log file per seed
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args.get("data_root"),
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top2": []}, {"top1": [], "top2": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        cnn_matrix.append(_grouped_row(cnn_accy))
        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top2"].append(cnn_accy["top2"])
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top2 curve: {}".format(cnn_curve["top2"]))

        if nme_accy is not None:
            logging.info("NME: {}".format(nme_accy["grouped"]))
            nme_matrix.append(_grouped_row(nme_accy))
            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top2"].append(nme_accy["top2"])
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top2 curve: {}\n".format(nme_curve["top2"]))
        else:
            logging.info("No NME accuracy.")

        logging.info("Average Accuracy (CNN): {}".format(np.mean(cnn_curve["top1"])))
        if nme_curve["top1"]:
            logging.info("Average Accuracy (NME): {}".format(np.mean(nme_curve["top1"])))

    for name, matrix in (("CNN", cnn_matrix), ("NME", nme_matrix)):
        if len(matrix) > 0:
            acctable, forgetting = _forgetting(matrix, task)
            logging.info("Accuracy Matrix ({}):\n{}".format(name, acctable))
            logging.info("Forgetting ({}): {}".format(name, forgetting))


def _grouped_row(accy):
    keys = sorted(k for k in accy["grouped"].keys() if "-" in k)
    return [accy["grouped"][k] for k in keys]


def _forgetting(matrix, task):
    """Same computation as the original PyCIL trainer (10x10 table for <=10 tasks)."""
    n_rows = max(10, len(matrix))
    n_cols = max(10, max(len(line) for line in matrix))
    acctable = np.zeros([n_rows, n_cols])
    for idxx, line in enumerate(matrix):
        acctable[idxx, : len(line)] = np.array(line)
    acctable = acctable.T
    forgetting = np.mean((np.max(acctable, axis=1) - acctable[:, task])[:task])
    return acctable, forgetting


def _set_device(args):
    gpus = []
    for device in args["device"]:
        if str(device) == "-1":
            gpus.append(torch.device("cpu"))
        else:
            gpus.append(torch.device("cuda:{}".format(device)))
    args["device"] = gpus


def _set_random():
    # As in the original code: torch seed is fixed to 1; the json "seed" only
    # controls the class order (see DataManager).
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
