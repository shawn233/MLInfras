'''
Author: shawn233
Date: 2021-04-02 15:13:26
LastEditors: shawn233
LastEditTime: 2021-04-02 16:04:08
Description: Export PyTorch model
'''

import os
import sys
import torch
import numpy as np
import pandas as pd

def export_resnet(model_path: str, output_path: str):
    ckpt = torch.load(model_path)
    print(
        f'epoch: {ckpt["epoch"]}, [train] loss: {ckpt["train_loss"]:.4f}, '
        f'acc: {100.*ckpt["train_acc"]:.2f}% [test] loss: {ckpt["test_loss"]:.4f} '
        f'acc: {100.*ckpt["test_acc"]:.2f}%')

    model = ckpt['model_state_dict']

    np.set_printoptions(precision=8, threshold=sys.maxsize)

    with open(output_path, "w") as out_f:
        for name in model:
            value = model[name].cpu().numpy()
            out_f.write(f"{name} {value.shape}\n")
            out_f.write(str(value) + "\n")
    


def main():
    export_resnet("./traffic/best.checker.ckpt", "./traffic/best.checker.txt")
    export_resnet("./traffic/best.task.ckpt", "./traffic/best.task.txt")
    # ckpt = torch.load("./iris/best.ckpt")
    # print(f'epoch: {ckpt["epoch"]}, training loss: {ckpt["train_loss"]:.4f}, '
    #         f'training acc: {100.*ckpt["train_acc"]:.2f}%')
    # model = ckpt["model_state_dict"]
    
    # with open("./iris/best.txt", "w") as f:
    #     for name in model:
    #         arr = model[name].numpy()
    #         var_name = name.split(".")[1].upper()
    #         s = [var_name+":"]
            
    #         if var_name == "WEIGHT":
    #             for i in range(len(arr)):
    #                 s.append("\t".join([str(w) for w in arr[i]]))
    #         elif var_name == "BIAS":
    #             s.append("\t".join([str(b) for b in arr]))
    #         else:
    #             raise ValueError(f"Unknown var_name {var_name}")

    #         # print (s)
    #         f.write("\n".join(s)+"\n")

    # print("Done!")


if __name__ == "__main__":
    main()