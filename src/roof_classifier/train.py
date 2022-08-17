import argparse
import logging
import sys
from math import prod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

from roof_classifier.dataset import AIRSDataset, SingleImage
from roof_classifier.model import RoofSegmenter
from roof_classifier.utils import get_device, read_tiff

# Configure logging
log_format = "%(asctime)s %(name)s %(levelname)s: %(message)s"
logging.basicConfig(stream = sys.stdout, 
                    filemode = "w",
                    format = log_format, 
                    level = logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    outputs_dir: Path,
    n_batches: int = 1,
    threshold: float = 0.5,
    device: str = "cpu",
):
    """Run validation on a model.

    Args:
        model (nn.Module): Model to run validation on
        val_loader (DataLoader): Validation data loader
        outputs_dir (Path): Directory in which to save images with validation output
        n_batches (int, optional): Number of batches to run validation on. Defaults to 1.
        threshold (float, optional): Threshold value for classification. Defaults to 0.5.
        device (str, optional): Device to use. Defaults to "cpu".
    """
    model.eval()

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        image = sample["image"].to(device)
        label = sample["label"].to(device)

        # Get model's classification output
        output = model(image).detach()
        output = (torch.sigmoid(output) > threshold).float()

        batch_size = image.shape[0]
        for i in range(batch_size):
            image_input = image[i]
            image_output = output[i].expand(3, -1, -1)
            image_label = label[i].expand(3, -1, -1)

            # Save the image, model output, and label as a TIFF image
            combined = torch.cat([image_input, image_output, image_label], dim=2)
            combined = transforms.ToPILImage()(combined)
            combined.save(outputs_dir / f"val_image_{batch_idx}_{i}.tiff")


def infer_image(
    model: nn.Module,
    image_path: Path,
    outputs_dir: Path,
    patch_size: int = 500,
    patch_stride: int = 250,
    threshold: float = 0.5,
    device: str = "cpu",
):
    """Run inference on a single image.

    Args:
        model (nn.Module): Model to use
        image_path (Path): Path to input image TIFF file
        outputs_dir (Path): Directory in which to save images with inference output
        patch_size (int, optional): Size of patches to feed to the model (in pixels).
            Defaults to 500.
        patch_stride (int, optional): Stride of patches to feed to the model (in pixels).
            Defaults to 250.
        threshold (float, optional): Threshold value for classification. Defaults to 0.5.
        device (str, optional): Device to use. Defaults to "cpu".
    """
    model.eval()

    # Read in image and initialize output tensor
    orig_image = read_tiff(image_path) / 255.0
    full_output = threshold + torch.zeros([1, orig_image.shape[1], orig_image.shape[2]])

    # Create data loader
    dataset = SingleImage(
        image_path,
        patch_size=patch_size,
        patch_stride=patch_stride,
    )
    loader = DataLoader(dataset, batch_size=1)

    for _, sample in enumerate(loader):
        i = sample["i"]
        j = sample["j"]
        image = sample["image"].to(device)

        # Get model's output for this patch
        output = model(image)
        output = torch.sigmoid(output.detach().cpu().squeeze())

        # For each pixel in this patch, replace the existing output
        # value with the new one if the confidence is higher (i.e.
        # the deviation from the threshold is larger)
        prev_output = full_output[
            0,
            i * patch_stride : i * patch_stride + patch_size,
            j * patch_stride : j * patch_stride + patch_size,
        ]
        mask = torch.abs(output - threshold) > torch.abs(prev_output - threshold)
        full_output[
            0,
            i * patch_stride : i * patch_stride + patch_size,
            j * patch_stride : j * patch_stride + patch_size,
        ][mask] = output[mask]

    # Get classification output for the full image
    full_output = (full_output > threshold).float().expand(3, -1, -1)

    # Save original image next to the model's output as a TIFF image
    combined = torch.cat([orig_image, full_output], dim=2)
    combined = transforms.ToPILImage()(combined)
    combined.save(outputs_dir / "test_image.tiff")


def train(
    n_epochs: int,
    dataset_info: dict,
    batch_size: int = 16,
    input_size: int = 512,
    n_val_batches: int = 1,
    threshold: float = 0.5,
    model_save_path: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
):
    """Run training pipeline.

    Args:
        n_epochs (int): Number of epochs to train for
        dataset_info (dict): Dictionary containing dataset paths
        batch_size (int, optional): Batch size to use for training. Defaults to 16.
        input_size (int, optional): Size of inputs to feed to the model during training
            and validation (in pixels). Defaults to 512.
        n_val_batches (int, optional): Number of batches to use for validation. Defaults to 1.
        threshold (float, optional): Threshold value for classification. Defaults to 0.5.
        model_save_path (Optional[Path], optional): Path to save model checkpoint.
            Defaults to None.
        outputs_dir (Optional[Path], optional): Directory in which to save validation and
            inference outputs. Defaults to None.
    """
    # Extract dataset information
    train_images_dir = Path(dataset_info["train_images_dir"])
    train_labels_dir = Path(dataset_info["train_labels_dir"])
    train_filenames_path = Path(dataset_info["train_filenames_path"])
    val_images_dir = Path(dataset_info["val_images_dir"])
    val_labels_dir = Path(dataset_info["val_labels_dir"])
    val_filenames_path = Path(dataset_info["val_filenames_path"])
    test_image_path = Path(dataset_info["test_image_path"])

    # Get device
    device = get_device()

    if outputs_dir is not None:
        outputs_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_set = AIRSDataset(
        train_images_dir,
        train_labels_dir,
        train_filenames_path,
        train=True,
        patch_size=1024,
        patch_stride=512,
    )
    val_set = AIRSDataset(
        val_images_dir,
        val_labels_dir,
        val_filenames_path,
        train=False,
        patch_size=input_size,
        patch_stride=input_size,
    )

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Initialize model
    model = RoofSegmenter().to(device)

    # Generate positive class weights for the loss function, based on the
    # percentage of positive class pixels seen in training data
    roof_ratios = []
    for batch_idx, sample in enumerate(train_loader):
        # Use a sample of 10 batches
        if batch_idx > 10:
            break
        roof_ratios.append(torch.sum(sample["label"]) / prod(sample["label"].shape))
    roof_ratio = np.mean(roof_ratios)
    pos_weight = (1 / roof_ratio) * torch.ones([input_size, input_size])

    # Create loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch+1}/{n_epochs}.")

        model.train()
        for batch_idx, sample in enumerate(train_loader):
            image = sample["image"].to(device)
            label = sample["label"].to(device)

            optimizer.zero_grad()

            # Get model output
            output = model(image)

            # Calculate loss & update weights
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            logger.info(f"Train loss: {loss.item()}")

            if outputs_dir is not None and batch_idx % 10 == 0:
                # Run inference on test image
                infer_image(
                    model,
                    test_image_path,
                    outputs_dir=outputs_dir,
                    threshold=threshold,
                    device=device,
                )

                # Run validation
                validate(
                    model,
                    val_loader,
                    outputs_dir=outputs_dir,
                    n_batches=n_val_batches,
                    threshold=threshold,
                    device=device,
                )

                if model_save_path is not None:
                    # Save model checkpoint
                    model_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SSD.")
    parser.add_argument(
        "--model_save_path",
        type=Path,
        default="checkpoint.pth",
        help="Where to save model checkpoints.",
    )
    parser.add_argument(
        "--outputs_dir",
        type=Path,
        default="./outputs",
        help="Directory in which to save model outputs.",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size to train with."
    )
    parser.add_argument(
        "--dataset_config",
        type=Path,
        default="./dataset.yaml",
        help="Path to YAML file containing dataset information.",
    )
    parser.add_argument(
        "--input_size", type=int, default=512, help="Size of model inputs (in pixels)."
    )
    args = parser.parse_args()

    # Load dataset YAML
    if not Path.exists(args.dataset_config):
        raise OSError(f"Dataset config YAML file {args.dataset_config} does not exist.")
    with open(args.dataset_config, "r") as f:
        dataset_info = yaml.safe_load(f)

    train(
        args.n_epochs,
        dataset_info,
        batch_size=args.batch_size,
        input_size=args.input_size,
        model_save_path=args.model_save_path,
        outputs_dir=args.outputs_dir,
    )
