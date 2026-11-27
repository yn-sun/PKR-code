import copy
import logging

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.utils.data import DataLoader

from utils.toolkit import accuracy, tensor2numpy

EPSILON = 1e-8
batch_size = 64
num_workers = 4  # loaders used for exemplar selection / class means (no shuffling)


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 2

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        assert self._total_classes != 0, "Total classes is 0"
        return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def after_task(self):
        pass

    def incremental_train(self, data_manager):
        raise NotImplementedError

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None
        return cnn_accy, nme_accy

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T
        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device)))
            else:
                _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))
            vectors.append(_vectors)
            targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)

    # ------------------------------------------------------------------ exemplars (herding)
    def _class_mean(self, data_manager, data, targets):
        dset = data_manager.get_dataset([], source="train", mode="test", appendent=(data, targets))
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        vectors, _ = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        return mean / np.linalg.norm(mean)

    def _herding(self, data_manager, class_idx, m):
        data, targets, idx_dataset = data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True
        )
        idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        vectors, _ = self._extract_vectors(idx_loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        class_mean = np.mean(vectors, axis=0)

        selected_exemplars, exemplar_vectors = [], []
        for k in range(1, m + 1):
            S = np.sum(exemplar_vectors, axis=0)
            mu_p = (vectors + S) / k
            i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
            selected_exemplars.append(np.array(data[i]))
            exemplar_vectors.append(np.array(vectors[i]))
            vectors = np.delete(vectors, i, axis=0)
            data = np.delete(data, i, axis=0)

        return np.array(selected_exemplars), np.full(m, class_idx)

    def _append_memory(self, data, targets):
        self._data_memory = (
            np.concatenate((self._data_memory, data)) if len(self._data_memory) != 0 else data
        )
        self._targets_memory = (
            np.concatenate((self._targets_memory, targets)) if len(self._targets_memory) != 0 else targets
        )

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets = copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._append_memory(dd, dt)
            self._class_means[class_idx, :] = self._class_mean(data_manager, dd, dt)

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            selected, targets = self._herding(data_manager, class_idx, m)
            self._append_memory(selected, targets)
            self._class_means[class_idx, :] = self._class_mean(data_manager, selected, targets)

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info("Constructing exemplars for new classes...({} per classes)".format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            _class_means[class_idx, :] = self._class_mean(
                data_manager, self._data_memory[mask], self._targets_memory[mask]
            )

        for class_idx in range(self._known_classes, self._total_classes):
            selected, targets = self._herding(data_manager, class_idx, m)
            self._append_memory(selected, targets)
            _class_means[class_idx, :] = self._class_mean(data_manager, selected, targets)

        self._class_means = _class_means
