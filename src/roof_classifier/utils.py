import logging

from PIL import Image
import cv2
import tifffile
import numpy as np
import torch


def read_tiff(file_path):
    return torch.Tensor(cv2.imread(str(file_path))).permute(2, 0, 1)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logging.info("Using CUDA.")
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("CUDA not available.")

    return device
