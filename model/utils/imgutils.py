'''
Author: shawn233
Date: 2021-01-15 17:54:21
LastEditors: shawn233
LastEditTime: 2021-01-17 18:37:08
Description: Image reading tools
'''
from PIL import Image
import numpy as np
import os
import logging
# import cv2
# import matplotlib.pyplot as plt
# from skimage import io



def _readImageFromFile(filepath, as_gray=False):
    '''
    Read an image from an **existing** file. Remark that this function
    does not consider invalid file path due to performance needs.

    Args:
    - filepath: str. A string representing a file path.
    - as_gray: Boolean. Set as True to convert the image to the gray scale.

    Returns: a numpy.ndarray object (H x W x C).
    '''

    with Image.open(filepath) as img:
        logging.info(f"Reading image {filepath}")
        if as_gray:
            img = img.convert("L")
            arr = np.asarray(img) # (H x W)
            arr = arr.reshape((*arr.shape, 1)) # add a channel dimension
        else:
            arr = np.asarray(img) # (H x W x C)
    return arr



def _readImageFromFileOrDir(file_or_dir, as_gray=False):
    '''
    Read an image from either a file or a directory.
    We also assume that in the given directory and all of its sub-directories,
    all files are image files so that Image.open(<fp>) can be directly invoked.
    
    Args:
    - file_or_dir: str. A string representing a file path or a directory.
    - as_gray: Boolean. Set as True to convert all images to the gray scale.

    Returns: a list of numpy.ndarray object [(H x W x C)].

    Notes:
    - The function currently skips file links (e.g. soft-link in Unix systems)
    '''
    
    if os.path.isfile(file_or_dir):
        arr = _readImageFromFile(file_or_dir, as_gray)
        return [arr]
    elif os.path.isdir(file_or_dir):
        ret = []
        for root, _, files in os.walk(file_or_dir, topdown=True):
            for filename in files:
                fp = os.path.join(root, filename)
                if fp.endswith(".lnk"):
                    logging.warning(f"Skipping {fp} because it is a Windows shortcut.")
                else:
                    ret.append(_readImageFromFile(fp, as_gray))
        return ret
    else:
        logging.warning(f"Skipping {file_or_dir} because it is neither a file path nor a directory.")




def readImages(files_or_dirs, as_gray=False):
    '''
    Read images from one of the following sources:
    - a single file: `readImages("./pics/lenna.png")`
    - a single directory: `readImages("./pics/folder")`
    - a list of files and directories: `readImages(["./pics/lenna.png", "./pics/folder"])`
    
    Args:
    - files_or_dirs: str or list<str>, locations where images will be read.
    - as_gray: Boolean. Set as True to convert all images to the gray scale.

    Returns: (N x H x W x C) numpy.ndarray object.
    '''

    if type(files_or_dirs) == type([]):
        # input is a list
        arr = []
        for obj in files_or_dirs:
            arr.extend(_readImageFromFileOrDir(obj, as_gray))
    elif type(files_or_dirs) == type(''):
        # input is a string
        arr = _readImageFromFileOrDir(files_or_dirs, as_gray)
    else:
        raise TypeError(f"Can not read from {files_or_dirs}, a {type(files_or_dirs)} object.")
    
    return np.stack(arr, axis=0)



def main():
    logging.basicConfig(level=logging.INFO)
    arr = readImages(["./pics/lenna.png", "./pics/folder"])
    print(arr.shape, type(arr))


if __name__ == "__main__":
    main()