'''
Author: shawn233
Date: 2021-01-18 21:44:58
LastEditors: shawn233
LastEditTime: 2021-04-10 11:03:07
Description: PyTorch training utils
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import time
import os
import shutil
import logging
from typing import Any, Optional, Callable, Union, List
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



def train(
        net, 
        dataset_train,
        dataset_test = None, 
        lr: float = 0.01,       # learning rate
        momentum: float = 0.9,      # for optim.SGD
        weight_decay: float = 0.01, # for SGD (0), Adam (0) and AdamW (0.01)
        reduce_lr_on_plateau: bool = False, # if true, use ReduceLROnPlateau lr scheduler
        plateau_factor: float = 0.1, # for ReduceLROnPlateau, `lr = lr * factor` when a metric plateaus 
        decay_delay: int = None,    # for StepLR (required argument): `lr` decay interval
        steplr_gamma: float = 0.1,  # for StepLR (0.1): `lr = lr * gamma` every `decay_delay` epochs
        batch_size: int = 128,
        epochs: int = 50,
        num_workers: int = None,   # (disabled) recommend: 4 x #GPU
        optimizer: str = "adam",
        device: str = "cpu",
        display_step: int = 100,
        model_root: str = None,     # if not None, save latest and best model to model_root
        best_only: bool = True,     # if True, only save the best checkpoint
        best_model_name: str = "best.ckpt", # name of the best model 
        save_interval: int = 10,    # interval to save model checkpoints, only effective when best_only is False
        load_latest: bool = False,  # if True, continue on checkpoint (model_latest.ckpt)
        # basic plotting utility
        plot_loss: bool = False,
        plot_acc: bool = False,
        loss_plot_path: str = "./loss_plot.png",
        acc_plot_path: str = "./acc_plot.png",
        plot_dump_path: str = "./plot.dmp",
        dpi: int = 100,
        drop_last: bool = False,    # for DataLoader
        early_stop_delay: int = 30, # if the best validation loss (if no val set, use train loss) 
                                    # is not updated for `early_stop_delay` epochs, then stop.
                                    # Set to None to disable the early-stop feature
        **kwargs
) -> None:
    logging.warning(f"Parameters not implemented: {kwargs}.")

    # Data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, 
                              shuffle=True, 
                              #num_workers=num_workers,
                              drop_last=drop_last,)
    if dataset_test is not None:
        test_loader = DataLoader(dataset_test, batch_size=batch_size,
                                 shuffle=False, 
                                 #num_workers=num_workers,
                                 drop_last=False
                                 )
    else:
        test_loader = None

    # Training device
    device = device.lower()
    if torch.cuda.is_available() and device == "cpu":
        logging.warning(f"Cuda is available but training device is CPU.")
    device = torch.device(device)
    net.to(device)

    # Optimizer
    optimizer = optimizer.lower()
    
    if optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {optimizer}.")
    
    if reduce_lr_on_plateau:
        # Reference: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=plateau_factor)
    elif decay_delay is not None:
        # Reference: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_delay,
                                            gamma=steplr_gamma)
    else:
        scheduler = None

    # Loss function
    criterion = nn.NLLLoss()

    # Plotting
    # n_batch_train = np.ceil(float(len(dataset_train)) / batch_size)
    # if drop_last:
    #     n_batch_train = n_batch_train - 1
    if plot_loss:
        train_loss_y, test_loss_y = [], []
    if plot_acc:
        train_acc_y, test_acc_y = [], []
    if plot_loss or plot_acc:
        train_x, test_x = [], []
        plot_x_cnt = 0

    # Model Saving
    best_model_acc = None
    best_model_loss = None
    train_loss, train_acc, test_loss, test_acc = None, None, None, None
    if model_root is not None:
        if not os.path.exists(model_root):
            logging.info(f"model_root not existed. Creating directory {model_root}")
            os.makedirs(model_root, exist_ok=False)

    # Result Reporting
    best_train_loss, best_train_acc = 1000000.0, 0.0
    best_test_loss, best_test_acc = 1000000.0, 0.0

    # Early Stop when best loss is updated for a long time
    best_unchanged_epochs: int = 0

    # Train by epochs
    for epoch in range(epochs):
        logging.info(
            f"epoch {epoch+1}/{epochs} "
            f"({early_stop_delay - best_unchanged_epochs} epochs to early stop)")
        net.train()
        # statistics: running statistics are cleared every `display_step` iterations
        #             train statistics are cleared every epoch
        running_loss = 0.0
        running_correct, running_total = 0, 0
        train_total_loss = 0.0
        train_total_correct, train_total = 0, 0

        # Train
        for idx, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # print(inputs.size())

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            # Suppose the model architecture has the last layer
            # z = self.fc_last(x)
            # then log_f = F.log_softmax(z), which applies to NLLLoss.
            log_fs = net(inputs)
            loss = criterion(log_fs, labels)
            loss.backward()
            optimizer.step()

            # statistics: loss
            running_loss += loss.item()
            train_total_loss += (loss.item() * labels.size(0))
            
            # statistics: accuracy
            running_total += labels.size(0)
            train_total += labels.size(0)
            _, predicted = torch.max(log_fs.data, 1)
            n_correct = (predicted == labels).sum().item()
            running_correct += n_correct
            train_total_correct += n_correct

            # plotting
            if plot_loss:
                train_loss_y.append(loss.item())
            if plot_acc:
                train_acc_y.append(n_correct / labels.size(0))
            if plot_loss or plot_acc:
                train_x.append(plot_x_cnt)
                plot_x_cnt += 1
            
            # show statistics
            if idx % display_step + 1 == display_step:
                logging.info(f"batch {idx+1}\t"
                             f"[train] loss: {running_loss / display_step:.4f}\t"
                             f"acc: {100. * running_correct / running_total:4.2f}%")
                running_loss = 0.0
                running_correct, running_total = 0, 0

        train_loss = train_total_loss / train_total
        train_acc = train_total_correct / train_total
        

        # Validate
        if test_loader is not None:
            test_total_loss = 0.0
            test_total_correct, test_total = 0, 0
            with torch.no_grad():
                net.eval()
                for idx, data in enumerate(test_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    log_fs = net(inputs)
                    loss = criterion(log_fs, labels)
                    
                    # statistics: loss
                    test_total_loss += (loss.item() * labels.size(0))
                    
                    # statistics: accuracy
                    test_total += labels.size(0)
                    _, predicted = torch.max(log_fs.data, 1)
                    test_total_correct += (predicted == labels).sum().item()

            # statistics: loss and accuracy
            test_loss = test_total_loss / test_total
            test_acc = test_total_correct / test_total
            
            best_unchanged_epochs += 1
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_unchanged_epochs = 0
            best_test_acc = max(best_test_acc, test_acc)

            # plotting
            if plot_loss:
                test_loss_y.append(test_loss)
            if plot_acc:
                test_acc_y.append(test_acc)
            if plot_loss or plot_acc:
                test_x.append(plot_x_cnt)

            logging.info(f"==> epoch {epoch+1}/{epochs}\t"
                        f"[train] loss: {train_loss:.4f}\tacc: {100.*train_acc:4.2f}%\t"
                        f"[test] loss: {test_loss:.4f}\tacc: {100.*test_acc:4.2f}%")
        else:
            # No validation dataset
            # For early stop, use train_loss
            best_unchanged_epochs += 1
            if train_loss < best_train_loss:
                best_unchanged_epochs = 0 # best_train_loss will be updated later

            logging.info(f"==> epoch {epoch+1}/{epochs}\t"
                        f"[train] loss: {train_loss:.4f}\tacc: {100.*train_acc:4.2f}%")
        
        best_train_loss = min(best_train_loss, train_loss)
        best_train_acc = max(best_train_acc, train_acc)

        if early_stop_delay is not None and best_unchanged_epochs >= early_stop_delay:
            logging.warning(
                f"Early Stop triggered at epoch {epoch}")
            break


        if scheduler is not None:
            if reduce_lr_on_plateau:
                scheduler.step(test_loss) #TODO: maybe use train_loss?
            elif decay_delay is not None:
                scheduler.step()
            else:
                raise RuntimeError("Unknown lr scheduler is used")    

        # Save
        if model_root is not None:
            try:
                if not best_only and epoch % save_interval == 0:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc
                    }, os.path.join(model_root, f"epoch{epoch}.ckpt"))

                # Check if the current model is better
                # Remark that the `best_model_acc` is different from 
                #   `best_test_acc` as the latter is used for plotting.
                save_best_flag = False
                if dataset_test is not None:
                    if best_model_acc is None:
                        best_model_acc = test_acc
                        best_model_loss = test_loss
                        save_best_flag = True
                    elif test_acc > best_model_acc:
                        best_model_acc = test_acc
                        save_best_flag = True
                    elif test_acc == best_model_acc:
                        if test_loss < best_model_loss:
                            best_model_loss = test_loss
                            save_best_flag = True
                else:
                    if best_model_acc is None:
                        best_model_acc = train_acc
                        best_model_loss = train_loss
                        save_best_flag = True
                    elif train_acc > best_model_acc:
                        best_model_acc = train_acc
                        save_best_flag = True
                    elif train_acc == best_model_acc:
                        if train_loss < best_model_loss:
                            best_model_loss = train_loss
                            save_best_flag = True
                
                if save_best_flag:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc
                    }, os.path.join(model_root, best_model_name))
                    logging.info(f"Best model updated at epoch {epoch} of "
                                f"model accuracy {100.*best_model_acc:4.2f}%")
            except PermissionError:
                logging.warning(f"Model (epoch {epoch}) not saved due to"
                                f" PermissionError. (model write too frequent)")

    logging.info("Training Finished.")
    logging.info(f"Report Best: "
                f"[train] loss: {best_train_loss:.4f} "
                f"acc: {100.*best_train_acc:4.2f}% "
                f"[test] loss: {best_test_loss:.4f} "
                f"acc: {100.*best_test_acc:4.2f}%")

    if plot_acc or plot_loss:
        plt_dump = {"train_x": train_x, "test_x": test_x}
        
        if plot_loss:
            plt_dump["train_loss"] = train_loss_y
            plt_dump["test_loss"] = test_loss_y
            
            plt.figure()
            plt.plot(train_x, train_loss_y, label="Train Loss")
            plt.plot(test_x, test_loss_y, label="Test Loss")
            plt.legend()
            plt.savefig(loss_plot_path, dpi=dpi)
        
        if plot_acc:
            plt_dump["train_acc"] = train_acc_y
            plt_dump["test_acc"] = test_acc_y
            
            plt.figure()
            plt.plot(train_x, train_acc_y, label="Train Accuracy")
            plt.plot(test_x, test_acc_y, label="Test Accuracy")
            plt.legend()
            plt.savefig(acc_plot_path, dpi=dpi)
        
        with open(plot_dump_path, "wb") as dmp:
            pickle.dump(plt_dump, dmp) 



def test(
        net,
        dataset_test,
        ckpt_path: str = None,          # If provided, load model from checkpoint
        device: str = "cpu",
        batch_size: int = 64,
        **kwargs
) -> None:
    logging.warning(f"Parameters not implemented: {kwargs}.")

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        net.load_state_dict(ckpt["model_state_dict"])
        logging.info(f"Checkpoint loaded: epoch {ckpt['epoch']} "
                    f"[train] loss: {ckpt['train_loss']:.4f} acc: {100.*ckpt['train_acc']:.2f}%"
                    f" [test] loss: {ckpt['test_loss']:.4f} acc: {100.*ckpt['test_acc']:.2f}%")

    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)

    # Testing device
    device = device.lower()
    if torch.cuda.is_available() and device == "cpu":
        logging.warning(f"Cuda is available but training device is CPU.")
    device = torch.device(device)
    net.to(device)

    # Loss function
    criterion = nn.NLLLoss()

    # Test 
    y_true, y_pred = [], []
    running_loss, running_total = 0.0, 0

    with torch.no_grad():
        net.eval()
        for idx, data in enumerate(test_loader, 0):
            y_true.append(data[1].numpy())
            inputs, labels = data[0].to(device), data[1].to(device)

            log_fs = net(inputs)
            loss = criterion(log_fs, labels)
            running_loss += loss
            running_total += labels.size(0)
            
            _, predicted = torch.max(log_fs.data, 1)
            y_pred.append(predicted.cpu().numpy())

    print("Testing Result:")
    print(classification_report(np.concatenate(y_true), np.concatenate(y_pred)))



def main():
    from model import LeNet5
    from datautils import MyMNIST
    from torchvision.datasets import MNIST
    from torchvision import transforms

    logging.basicConfig(level=logging.INFO)
    net = LeNet5(mnist=True)
    transform = transforms.ToTensor()
    dataset_train = MyMNIST("./dataset/mnist/", train=True, transform=transform, download=False)
    dataset_test = MyMNIST("./dataset/mnist/", train=False, transform=transform, download=False)
    params = {
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "gamma": 0.1,
        "reduce_lr_on_plateau": False,
        "decay_delay": None,
        "batch_size": 128,
        "epochs": 5,
        "early_stop_delay": 30,
        "optimizer": "adam",
        "device": "cuda:0",
        "display_step": 100,
        "model_root": None,
        "best_only": True,
        "best_model_name": "best.ckpt",
        "save_interval": 10,
        "load_latest": False,
    }
    train(net, dataset_train, dataset_test, **params)



if __name__ == "__main__":
    main()