from itertools import chain, cycle
from math import prod
from pathlib import Path
from typing import Iterator

import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, IterableDataset

from roof_classifier.transforms import (RandomCrop, RandomHorizontalFlip,
                                        RandomRotation, RandomVerticalFlip)
from roof_classifier.utils import read_tiff


class AIRSDataset(IterableDataset):
    """Class to load in patches from images and labels in the
    AIRS dataset, which can be found here: https://www.airs-dataset.com
    """
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        names_file: Path,
        patch_size: int = 1000,
        patch_stride: int = 500,
        crop_size: int = 512,
        max_rotation_degrees: float = 45,
        p_horizontal_flip: float = 0.2,
        p_vertical_flip: float = 0.2,
        train: bool = False,
        min_roof_ratio: float = 0.05,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.min_roof_ratio = min_roof_ratio
        self.train = train

        # Create list of filenames for images and labels
        with open(names_file, "r") as f:
            self.names = f.readlines()
            self.names = [line.rstrip() for line in self.names]
            if not self.train:
                # Add extension to validation set filenames
                self.names = [line + ".tif" for line in self.names]

        if self.train:
            self.transform = transforms.Compose(
            [
                RandomRotation(max_rotation_degrees),
                RandomCrop(crop_size),
                RandomHorizontalFlip(p_horizontal_flip),
                RandomVerticalFlip(p_vertical_flip),
            ]
            )
        else:
            self.transform = None

    def get_patches(self, name: str) -> Iterator[dict]:
        """Generate all valid patches of an image and label pair.

        Args:
            name (str): Filename of image and label

        Yields:
            Iterator[dict]: Iterator that yields a sample for each valid
                patch in the image
        """
        image_path = self.image_dir / name
        label_path = self.label_dir / name
        image = read_tiff(image_path) / 255.0
        label = read_tiff(label_path)
        label = label[0].unsqueeze(0)

        # Generate patches from the image and label
        image_patches = image.unfold(1, self.patch_size, self.patch_stride).unfold(
            2, self.patch_size, self.patch_stride
        )
        label_patches = label.unfold(1, self.patch_size, self.patch_stride).unfold(
            2, self.patch_size, self.patch_stride
        )

        for i in range(image_patches.shape[1]):
            for j in range(image_patches.shape[2]):
                image_patch = image_patches[:, i, j, :, :]
                label_patch = label_patches[:, i, j, :, :]

                if self.train:
                    # For training, ignore patches without a minimum roof pixel ratio
                    roof_ratio = torch.sum(label_patch)/prod(label_patch.shape)
                    if roof_ratio < self.min_roof_ratio:
                        continue
                
                # Apply transforms
                sample = {"image": image_patch, "label": label_patch}
                if self.transform:
                    sample = self.transform(sample)

                if self.train:
                    # For training, ignore patches without a minimum roof pixel ratio
                    roof_ratio = torch.sum(sample["label"])/prod(sample["label"].shape)
                    if roof_ratio < self.min_roof_ratio:
                        continue

                yield sample

    def __iter__(self) -> dict:
        """Loop through each filename and iterate through every patch of the
        corresponding sample.

        Returns:
            dict: Sample containing image and label patch
        """
        return chain.from_iterable(map(self.get_patches, cycle(self.names)))


class SingleImage(IterableDataset):
    """Class to iterate through a single TIFF image and produce
    patches that cover the entire image.
    """
    def __init__(
        self,
        image_path: Path,
        patch_size: int = 512,
        patch_stride: int = 256,
    ):
        self.image_path = image_path
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def get_patches(self, name: str) -> Iterator[dict]:
        """Generate all patches of an image.

        Args:
            name (str): Filename of the image

        Yields:
            Iterator[dict]: Iterator that yields a sample for each
                patch in the image, including the coordinates of the
                top-left corner of the patch
        """
        image_path = name
        image = read_tiff(image_path) / 255.0

        image_patches = image.unfold(1, self.patch_size, self.patch_stride).unfold(
            2, self.patch_size, self.patch_stride
        )

        for i in range(image_patches.shape[1]):
            for j in range(image_patches.shape[2]):
                image_patch = image_patches[:, i, j, :, :]

                sample = {"image": image_patch, "i": i, "j": j}

                yield sample

    def __iter__(self) -> dict:
        """Iterate through every patch of the desired image.

        Returns:
            dict: Sample containing image and label patch
        """
        return chain.from_iterable(map(self.get_patches, [self.image_path]))


if __name__ == "__main__":
    dataset_config = Path("./dataset.yaml")

    # Load dataset YAML
    if not Path.exists(dataset_config):
        raise OSError(f"Dataset config YAML file {dataset_config} does not exist.")
    with open(dataset_config, "r") as f:
        dataset_info = yaml.safe_load(f)

    dataset = AIRSDataset(
        Path(dataset_info["train_images_dir"]),
        Path(dataset_info["train_labels_dir"]),
        Path(dataset_info["train_filenames_path"]),
        train=True
    )

    debug_path = Path("./debug")
    debug_path.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size)

    # Save images and labels from a batch
    for batch_idx, sample in enumerate(loader):
        if batch_idx > 0:
            break

        for i in range(batch_size):
            image = sample["image"][i]
            image = transforms.ToPILImage()(image)
            image.save(debug_path / f"debug_image_{i}.tiff")

            label = sample["label"][i]
            label = transforms.ToPILImage()(label)
            label.save(debug_path / f"debug_label_{i}.tiff")
