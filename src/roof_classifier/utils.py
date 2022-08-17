import logging
from pathlib import Path

import cv2
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def read_tiff(file_path: Path) -> torch.Tensor:
    """Load a TIFF image as a tensor.

    Args:
        file_path (Path): Path to TIFF tile

    Returns:
        torch.Tensor: 3 x H x W tensor containing image data
    """
    return torch.Tensor(cv2.imread(str(file_path))).permute(2, 0, 1)


def get_device() -> str:
    """Get best device for running a model.

    Returns:
        str: Device name
    """
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using {device} backend.")

    return device
