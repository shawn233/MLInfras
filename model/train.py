'''
Author: shawn233
Date: 2021-04-01 03:48:28
LastEditors: shawn233
LastEditTime: 2021-04-10 11:03:56
Description: Train model
'''

import os
import argparse
import logging
from typing import Any, Optional, Callable, Union, List, Tuple
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import joblib
from torchvision.transforms.transforms import ToTensor

from model import ResNet34
from utils.trainutils import train, test
from utils.datautils import SmallDataset


class Iris(SmallDataset):

    def __init__(
            self,
            root: str,
            train: bool = True,
            train_ratio: float = 0.8,
    ) -> None:
        super(Iris, self).__init__(root, train)

        features = []
        with open(os.path.join(self.root, "iris-features.txt")) as f:
            for line in f:
                line = line.strip()
                features.append([float(i) for i in line.split('\t')])
        
        _data = torch.tensor(features)
        print(_data.dtype, _data.shape)

        labels_ = []
        with open(os.path.join(self.root, "iris-labels.txt")) as f:
            for line in f:
                line = line.strip()
                labels_.append(int(line))

        _labels = torch.tensor(labels_)
        print(_labels.dtype, _labels.shape)

        train_ind = int(train_ratio * len(_labels))
        if train:
            self.data = _data[:train_ind]
            self.labels = _labels[:train_ind]
        else:
            self.data = _data[train_ind:]
            self.labels = _labels[train_ind:]



class GTSRB(SmallDataset):

    def __init__(
            self,
            root: str = "../data/traffic/",
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            dump_root: str = "./dump/",
            force_reload: bool = True, # force reload
    ) -> None:
        super(GTSRB, self).__init__(root, train, transform, target_transform, False)

        # self.root = root
        # self.transform = transform
        # self.target_transform = target_transform

        # # load data from `self.root`, depending on the value of `train`
        # self.data = None
        # self.labels = None

        tf_resize = transforms.Compose([transforms.ToTensor(), transforms.Resize((96, 96))])

        os.makedirs(dump_root, exist_ok=True)

        if train:
            # get the training dataset
            dump_path = os.path.join(dump_root, "gtsrb-train.npz")
            if not force_reload and os.path.exists(dump_path):
                logging.info(f"[GTSRB] Loading training dataset from {dump_path} ...")
                npzfile = np.load(dump_path, allow_pickle=True)
                #data_dict = torch.load(dump_path)
                _data = npzfile["_data"]
                _labels = npzfile["_labels"]
            else:
                logging.info("[GTSRB] Reading training dataset ...")
                # preliminary: 43 classes
                n_classes: int = 43
                _data, _labels = [], []

                for label in range(n_classes):
                    label_root = os.path.join(
                        self.root, "GTSRB", "Final_Training", "Images", f"{label:05}")
                    label_df = pd.read_csv(
                        os.path.join(label_root, f"GT-{label:05}.csv"),
                        delimiter=';')
                    logging.info(f"[GTSRB] Reading class {label} ({label_df.shape[0]} samples)")
                    for i in range(label_df.shape[0]):
                        image_path = os.path.join(label_root, label_df.loc[i, 'Filename'])
                        with Image.open(image_path) as img:
                            _data.append(np.asarray(tf_resize(img)))
                        _labels.append(label)

                #torch.save({"_data":_data, "_labels":_labels}, dump_path)
                np.savez(dump_path, _data=_data, _labels=_labels)
        else:
            # get the testing dataset (as validation dataset)
            dump_path = os.path.join(dump_root, "gtsrb-test.npz")
            if not force_reload and os.path.exists(dump_path):
                logging.info(f"[GTSRB] Loading testing dataset from {dump_path} ...")
                npzfile = np.load(dump_path, allow_pickle=True)
                #data_dict = torch.load(dump_path)
                _data = npzfile["_data"]
                _labels = npzfile["_labels"]
            else:
                logging.info("[GTSRB] Reading testing dataset ...")
                _data, _labels = [], []
                test_df = pd.read_csv(
                    os.path.join(self.root, "GT-final_test.csv"), 
                    delimiter=';')
                test_root = os.path.join(self.root, "GTSRB", "Final_Test", "Images")
                for i in range(test_df.shape[0]):
                    if i != 0 and i % 1000 == 0:
                        logging.info(f"[GTSRB] Reading testing sample {i}/{test_df.shape[0]}")
                    image_path = os.path.join(test_root, test_df.loc[i, 'Filename'])
                    with Image.open(image_path) as img:
                        _data.append(np.asarray(tf_resize(img)))
                    _labels.append(test_df.loc[i, 'ClassId'])
                np.savez(dump_path, _data=_data, _labels=_labels)

        self.data = torch.from_numpy(np.asarray(_data)).permute(0, 3, 1, 2).float()
        self.labels = torch.from_numpy(np.asarray(_labels)).long()

        print(f"GTSRB {'train' if train else 'test'} dataset "
              f"images: {self.data.size()}, labels: {self.labels.size()}")


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        sample = self.data[idx]
        target = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        if target is not None:
            if self.target_transform is not None:
                target = self.target_transform(target)
        
        return sample, target


    def __len__(self) -> int:
        return len(self.labels)



def set_seeds(s: int = 703, set_cudnn: bool = False):
    import torch
    import random
    import numpy as np

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checker", action="store_true")
    args = parser.parse_args()
    
    PLOT_ROOT = "./figures/"
    os.makedirs(PLOT_ROOT, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s", 
        datefmt="%Y/%m/%d %H:%M:%S")

    set_seeds()

    # checker_net = ResNet34(43, 0.15)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    # ])
    # transform = transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    transform = transforms.ToTensor()
    if not args.checker:
        logging.info("Training task model ...")
        task_net = ResNet34(43)
        traffic = GTSRB(transform=transform, force_reload=False)
        traffic_test = GTSRB(train=False, transform=transform, force_reload=False)
        #exit()
        params = {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "reduce_lr_on_plateau": True,
            "batch_size": 256,
            "epochs": 500,
            "early_stop_delay": 50,
            "optimizer": "sgd",
            "device": "cuda:0",
            "display_step": 10,
            "model_root": "./traffic/",
            "best_only": True,
            "best_model_name": "best.task.ckpt",
            "load_latest": False,
            "plot_loss": True,
            "plot_acc": True,
            "loss_plot_path": os.path.join(PLOT_ROOT, "task_loss.png"),
            "acc_plot_path": os.path.join(PLOT_ROOT, "task_acc.png"),
            "plot_dump_path": os.path.join(PLOT_ROOT, "task_plot.dmp"),
        }
        train(task_net, traffic, traffic_test, **params)
    else:
        logging.info("Training checker model ...")
        checker_net = ResNet34(43, 0.15)
        traffic = GTSRB(transform=None, force_reload=False)
        traffic_test = GTSRB(train=False, transform=None, force_reload=False)
        #exit()
        params = {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "reduce_lr_on_plateau": True,
            "batch_size": 256,
            "epochs": 200,
            "early_stop_delay": 30,
            "optimizer": "sgd",
            "device": "cuda:0",
            "display_step": 10,
            "model_root": "./traffic/",
            "best_only": True,
            "best_model_name": "best.checker.ckpt",
            "load_latest": False,
            "plot_loss": True,
            "plot_acc": True,
            "loss_plot_path": os.path.join(PLOT_ROOT, "checker_loss.png"),
            "acc_plot_path": os.path.join(PLOT_ROOT, "checker_acc.png"),
            "plot_dump_path": os.path.join(PLOT_ROOT, "checker_plot.dmp"),
        }
        train(checker_net, traffic, traffic_test, **params)


def detailed_eval():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s", 
        datefmt="%Y/%m/%d %H:%M:%S")
    transform = None
    traffic_test = GTSRB(train=False, transform=transform, force_reload=False)
    test_loader = DataLoader(traffic_test, batch_size=1, 
                              shuffle=False, 
                              #num_workers=num_workers,
                              drop_last=False,)
    test_iter = test_loader._get_iterator()
    data, labels = test_iter.next()

    net = ResNet34(43, 0.15)
    model_dict = torch.load("./traffic/best.checker.ckpt")
    model_state_dict = model_dict["model_state_dict"]
    net.load_state_dict(model_state_dict)

    print(data)
    print(labels)

    with torch.no_grad():
        net.eval()
        net(data)
        log_fs = net(data)
        _, predicted = torch.max(log_fs.data, 1)
        # print(fs)
        print(predicted)



if __name__ == "__main__":
    main()
    # detailed_eval()