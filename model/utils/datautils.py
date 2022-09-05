'''
Author: shawn233
Date: 2021-01-17 19:21:41
LastEditors: shawn233
LastEditTime: 2021-04-06 11:21:06
Description: PyTorch dataset utils
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import numpy as np
import logging

from typing import Any, Callable, List, Optional, Union, Tuple




class CustomDataset(Dataset):
    '''
    In custom dataset, we always load all labels into memory.
    The size of the dataset is always the length of labels.
    '''


    def show(self, nrow:int = 8, ncol: int = 8, save_path: str = None, 
            by_label: bool = False, **kwargs) -> None:
        if by_label:
            raise Exception(f"`by_label` argument not implemented. Try to set `by_label=False`.")
        else:
            # randomly obtain a batch of images
            indices = np.arange(len(self))
            np.random.shuffle(indices)

            n_images = min(nrow * ncol, len(self))
            grid_images = [self[indices[i]][0] for i in range(n_images)] # a list of tensors
            grid_images = torch.stack(grid_images, dim=0)
            
            # show grid image using PIL
            grid_image = make_grid(grid_images, nrow=nrow, **kwargs)
            img = to_pil_image(grid_image)
            img.show()

            if save_path is not None:
                logging.info(f"Saving grid image to {save_path}")
                save_image(grid_images, nrow=nrow, **kwargs)


    def info(self):
        pass




class SmallDataset(CustomDataset):
    '''
    In small datasets, we directly load all data samples into memory.
    '''

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
    
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # load data from `self.root`, depending on the value of `train`
        self.data = None
        self.labels = None

        # _data, _labels = [], []

        # with open(os.path.join(self.root, "data.txt"), "r") as f:
        #     for line in f:
        #         line = line.strip().split()
        #         _data.append([float(i) for i in line[:-1]])
        #         _labels.append(int(float(line[-1])))

        # train_size = int(0.7 * len(_labels))
        # indices = np.arange(len(_labels))
        # np.random.shuffle(indices)

        # _data = np.asarray(_data)
        # _labels = np.asarray(_labels)

        # if train:
        #     self.data = _data[indices[:train_size]]
        #     self.labels = _labels[indices[:train_size]]
        # else:
        #     self.data = _data[indices[train_size:]]
        #     self.labels = _labels[indices[train_size:]]

        # print(self.data.shape, self.labels.shape)


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        sample = self.data[idx, :]
        target = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        if target is not None:
            if self.target_transform is not None:
                target = self.target_transform(target)
        
        return sample, target


    def __len__(self) -> int:
        return len(self.labels)



class LargeDataset(CustomDataset):
    '''
    In larget datasets, we delay the data loading process to access time.
    '''

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
    
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # load data from `self.root`, depending on the value of `train`
        self.labels = None


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        sample = None # load from local storage
        target = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        if target is not None:
            if self.target_transform is not None:
                target = self.target_transform(target)
        
        return sample, target


    def __len__(self) -> int:
        return len(self.labels)



def main():
    from torchvision.datasets import MNIST

    class MyMNIST(MNIST):
        def show(self, nrow:int = 8, ncol: int = 8, save_path: str = None, 
                by_label: bool = False, **kwargs) -> None:
            if by_label:
                raise Exception(f"`by_label` argument not implemented. Try to set `by_label=False`.")
            else:
                # randomly obtain a batch of images
                indices = np.arange(len(self))
                np.random.shuffle(indices)

                n_images = min(nrow * ncol, len(self))
                grid_images = [self[indices[i]][0] for i in range(n_images)] # a list of tensors
                grid_images = torch.stack(grid_images, dim=0)
                
                # show grid image using PIL
                grid_image = make_grid(grid_images, nrow=nrow, **kwargs)
                img = to_pil_image(grid_image)
                img.show()

                if save_path is not None:
                    logging.info(f"Saving grid image to {save_path}")
                    save_image(grid_images, nrow=nrow, **kwargs)

    logging.basicConfig(level=logging.INFO)
    transform = transforms.ToTensor()
    my_mnist = MyMNIST("./dataset/mnist", train=True, download=True, transform=transform)
    my_mnist.show(nrow=10, ncol=10, padding=10, pad_value=1.0)


if __name__ == "__main__":
    main()