import argparse
import logging
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


def validate(model: nn.Module, val_loader, outputs_dir: Path, device: str = "cpu"):
    model.eval()

    for batch_idx, sample in enumerate(val_loader):
        if batch_idx > 0:
            break

        image = sample["image"].to(device)
        label = sample["label"].to(device)

        batch_size = image.shape[0]
        output = model(image).detach()

        output = (torch.sigmoid(output) > 0.5).float()

        for i in range(batch_size):
            image_output = output[i].expand(3, -1, -1)
            image_input = image[i]
            image_label = label[i].expand(3, -1, -1)
            combined = torch.cat([image_input, image_output, image_label], dim=2)
            combined = transforms.ToPILImage()(combined)
            combined.save(outputs_dir / f"val_image_{i}.tiff")


def infer_image(
    model: nn.Module,
    image_path: Path,
    outputs_dir: Path,
    device: str = "cpu",
    patch_size: int = 500,
    patch_stride: int = 250,
):
    model.eval()

    orig_image = read_tiff(image_path) / 255.0
    pred = 0.5 + torch.zeros([1, orig_image.shape[1], orig_image.shape[2]])
    pred = pred

    dataset = SingleImage(
        image_path,
        patch_size=patch_size,
        patch_stride=patch_stride,
    )
    loader = DataLoader(dataset, batch_size=1)

    for batch_idx, sample in enumerate(loader):
        i = sample["i"]
        j = sample["j"]
        image = sample["image"].to(device)

        output = model(image)
        output = torch.sigmoid(output.detach().cpu().squeeze())
        current_pred = pred[
            0,
            i * patch_stride : i * patch_stride + patch_size,
            j * patch_stride : j * patch_stride + patch_size,
        ]
        mask = torch.abs(output - 0.5) > torch.abs(current_pred - 0.5)
        pred[
            0,
            i * patch_stride : i * patch_stride + patch_size,
            j * patch_stride : j * patch_stride + patch_size,
        ][mask] = output[mask]


    full_output = (pred > 0.5).float().expand(3, -1, -1)

    combined = torch.cat([orig_image, full_output], dim=2)
    combined = transforms.ToPILImage()(combined)
    combined.save(outputs_dir / "test_image.tiff")


def train(
    n_epochs: int,
    dataset_info: dict,
    batch_size: int = 16,
    input_size: int = 512,
    model_save_path: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
):
    # Extract dataset information
    train_images_dir = Path(dataset_info["train_images_dir"])
    train_labels_dir = Path(dataset_info["train_labels_dir"])
    train_filenames_path = Path(dataset_info["train_filenames_path"])
    val_images_dir = Path(dataset_info["val_images_dir"])
    val_labels_dir = Path(dataset_info["val_labels_dir"])
    val_filenames_path = Path(dataset_info["val_filenames_path"])

    # Get device
    device = get_device()

    if outputs_dir is not None:
        outputs_dir.mkdir(parents=True, exist_ok=True)

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

    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

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

    criterion = nn.BCEWithLogitsLoss(pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch+1}/{n_epochs}.")

        model.train()
        for batch_idx, sample in enumerate(train_loader):
            image = sample["image"].to(device)
            label = sample["label"].to(device)

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print(loss.item())

            if outputs_dir is not None and batch_idx % 10 == 0:
                infer_image(
                    model,
                    Path(
                        "/Users/jordan/Documents/Work/Job Search/Invision AI/Recruiting exercise ML/image.tif"
                    ),
                    outputs_dir=outputs_dir,
                    device=device,
                )
                validate(model, val_loader, outputs_dir=outputs_dir, device=device)
        
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
