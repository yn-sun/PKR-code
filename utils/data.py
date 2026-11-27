import os

import numpy as np
from torchvision import datasets, transforms

from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    use_path = False


class iCIFAR10(iData):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]
    class_order = np.arange(10).tolist()

    def __init__(self, root="./data"):
        self.root = root

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(self.root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(self.root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]
    class_order = np.arange(100).tolist()

    def __init__(self, root="./data"):
        self.root = root

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(self.root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(self.root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iImageNet200(iData):
    """TinyImageNet-200 in its original train/val directory layout."""

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(200).tolist()

    def __init__(self, root="./data/tiny-imagenet-200"):
        self.root = root

    def download_data(self):
        train_dir = os.path.join(self.root, "train")
        val_dir = os.path.join(self.root, "val")
        val_images_dir = os.path.join(val_dir, "images")
        val_annotations = os.path.join(val_dir, "val_annotations.txt")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"TinyImageNet train directory not found: {train_dir}")
        if not os.path.isdir(val_images_dir):
            raise FileNotFoundError(f"TinyImageNet validation images directory not found: {val_images_dir}")
        if not os.path.isfile(val_annotations):
            raise FileNotFoundError(f"TinyImageNet validation annotation file not found: {val_annotations}")

        train_dset = datasets.ImageFolder(train_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)

        class_to_idx = train_dset.class_to_idx
        test_data, test_targets = [], []
        with open(val_annotations, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                image_name, wnid = fields[:2]
                image_path = os.path.join(val_images_dir, image_name)
                if wnid in class_to_idx and os.path.isfile(image_path):
                    test_data.append(image_path)
                    test_targets.append(class_to_idx[wnid])

        if not test_data:
            raise RuntimeError("No TinyImageNet validation samples were parsed.")
        self.test_data = np.asarray(test_data)
        self.test_targets = np.asarray(test_targets, dtype=np.int64)


class iPathMNIST(iData):
    train_trsf = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [transforms.Normalize(mean=[0.5], std=[0.5])]
    class_order = np.arange(9).tolist()

    def __init__(self, root=None):
        self.root = root

    def download_data(self):
        from medmnist import PathMNIST  # optional dependency

        kwargs = {"root": self.root} if self.root else {}
        train_dataset = PathMNIST(split="train", download=True, **kwargs)
        test_dataset = PathMNIST(split="test", download=True, **kwargs)
        self.train_data, self.train_targets = train_dataset.imgs, np.array(train_dataset.labels).reshape(-1)
        self.test_data, self.test_targets = test_dataset.imgs, np.array(test_dataset.labels).reshape(-1)
