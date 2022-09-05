'''
Author: shawn233
Date: 2021-04-01 03:53:14
LastEditors: shawn233
LastEditTime: 2021-04-01 20:20:43
Description: Models
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import *
import numpy as np


class TwoLayerFC(nn.Module):
    
    def __init__(self, *args):
        super(TwoLayerFC, self).__init__() # necessary otherwise error
        self.fc = nn.Linear(in_features=4, out_features=3, bias=True)    

    def forward(self, x):
        z = self.fc(x)
        log_fs = F.log_softmax(z, dim=1)
        
        return log_fs


class BuildingBlock(nn.Module):
    
    def __init__(
        self, 
        kernels: List[int], 
        in_channels: int, 
        out_channels: int,
    ):
        super(BuildingBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        _layers = []

        start_idx = 0
        self.shortcut = None # linear projection matrix W_s in the paper

        if in_channels != out_channels: # this means we need a downsampling
            start_idx = 1

            if in_channels * 2 != out_channels:
                raise ValueError(
                    f"in_channels {in_channels}, out_channels {out_channels}")
            if kernels[0] % 2 == 0:
                raise ValueError(f"Kernel size {kernels[0]}")
            
            # linear projection if the dimension increases
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, 1, stride=2, padding=0, bias=False)

            _layers.append((
                f"conv1", 
                nn.Conv2d(
                    in_channels, out_channels, kernels[0], stride=2,
                    padding=((kernels[0] - 1) // 2), padding_mode='zeros',
                    bias=False)))
            _layers.append((f"bn1", nn.BatchNorm2d(out_channels)))
            if len(kernels) > 1:
                _layers.append((f"relu1", nn.ReLU(inplace=True)))

        for idx, kernel_size in enumerate(kernels[start_idx:], start_idx):
            if kernel_size % 2 == 0:
                raise ValueError(f"Kernel size {kernel_size}")
            _layers.append((
                f"conv{idx+1}", 
                nn.Conv2d(
                    out_channels, out_channels, kernel_size, stride=1,
                    padding=((kernel_size - 1) // 2), padding_mode='zeros',
                    bias=False)))
            _layers.append((f"bn{idx+1}", nn.BatchNorm2d(out_channels)))
            if idx + 1 < len(kernels):
                _layers.append((f"relu{idx+1}", nn.ReLU(inplace=True)))

        self.layers = nn.Sequential(OrderedDict(_layers))


    def forward(self, x:torch.Tensor):
        out = self.layers(x)
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)
        out = out + residual
        return F.relu(out)



class ResNet34(nn.Module):

    def __init__(self, n_classes:int, alpha:float=1.0):
        '''
        Args:
        - n_classes: number of categories, used to determine the output layer
        - alpha: used to scale the network
        '''
        super(ResNet34, self).__init__()

        def __repeat_blocks(
            name:str,  # e.g. "conv2"
            kernels:List[int],
            in_channels:int, 
            out_channels:int,
            n_repeat:int
        ):
            _layers = []
            _layers.append(
                (f"{name}_1", BuildingBlock(kernels, in_channels, out_channels)))
            for i in range(1, n_repeat):
                _layers.append(
                    (f"{name}_{i+1}", 
                     BuildingBlock(kernels, out_channels, out_channels)))
            return _layers

        c0 = np.round(64 * alpha).astype(int) # initial channel
        n_channels = [c0, 2*c0, 4*c0, 8*c0]

        self.resnet = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    3, n_channels[0], 7, stride=2, 
                    padding=3, padding_mode='zeros', bias=False)),
                ('bn', nn.BatchNorm2d(n_channels[0])),
                ('relu', nn.ReLU(inplace=True)),
            ]))), 
            ('maxpool', nn.MaxPool2d(3, stride=2, padding=1)),
            *__repeat_blocks("conv2", [3, 3], n_channels[0], n_channels[0], 3),
            *__repeat_blocks("conv3", [3, 3], n_channels[0], n_channels[1], 4),
            *__repeat_blocks("conv4", [3, 3], n_channels[1], n_channels[2], 6),
            *__repeat_blocks("conv5", [3, 3], n_channels[2], n_channels[3], 3),
            ('gap', nn.AdaptiveAvgPool2d(1)),
        ]))

        self.fc = nn.Linear(n_channels[3], n_classes, bias=True)


    def forward(self, x):
        # For debug
        # z = self.resnet[0](x)
        
        # print("forwarding in module 3 (maxpool) ...")
        # z = self.resnet[1](z)
        # print(z)

        # layer_idx = 2
        # print("forwarding in module 4 (conv2) ...")
        # for _ in range(3):
        #     z = self.resnet[layer_idx](z)
        #     layer_idx += 1
        # print(z)

        # print("forwarding in module 5 (conv3) ...")
        # for _ in range(4):
        #     z = self.resnet[layer_idx](z)
        #     layer_idx += 1
        # print(z)

        # print("forwarding in module 6 (conv4) ...")
        # for _ in range(6):
        #     z = self.resnet[layer_idx](z)
        #     layer_idx += 1
        # print(z)

        # print("forwarding in module 7 (conv5) ...")
        # for _ in range(3):
        #     z = self.resnet[layer_idx](z)
        #     layer_idx += 1
        # print(z)

        # print("forwarding in module 8 (gap) ...")
        # z = self.resnet[layer_idx](z)
        # print(z)

        z = self.resnet(x)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        log_fs = F.log_softmax(z, dim=1)
        # fs = F.softmax(z, dim=1)

        return log_fs



def main():
    resnet = ResNet34(43)
    print(resnet)

    # from torchvision import models
    # print(models.resnet34())


if __name__ == "__main__":
    main()