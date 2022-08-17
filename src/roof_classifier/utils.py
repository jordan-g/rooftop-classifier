import logging

import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logging.info("Using CUDA.")
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("CUDA not available.")

    return device
