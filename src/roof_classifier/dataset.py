from math import prod
import random
from itertools import chain, cycle
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop, hflip, rotate, vflip
from torch.utils.data import IterableDataset, DataLoader

from roof_classifier.utils import read_tiff


class AIRSDataset(IterableDataset):
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        names_file: Path,
        patch_size: int = 1000,
        patch_stride: int = 500,
        train: bool = False,
        min_roof_ratio: float = 0.05,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.min_roof_ratio = min_roof_ratio
        self.train = train

        with open(names_file, "r") as f:
            self.names = f.readlines()
            self.names = [line.rstrip() for line in self.names]
            if not self.train:
                # Add extension to validation set filenames
                self.names = [line + ".tif" for line in self.names]

        if self.train:
            self.transform = transforms.Compose(
            [
                RandomRotation(45),
                RandomCrop(512),
                RandomHorizontalFlip(0.2),
                RandomVerticalFlip(0.2),
            ]
            )
        else:
            self.transform = None

    def process_sample(self, name):
        image_path = self.image_dir / name
        label_path = self.label_dir / name
        image = read_tiff(image_path) / 255.0
        label = read_tiff(label_path)
        label = label[0].unsqueeze(0)

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

                # if self.train:
                # For training, ignore patches without a minimum roof pixel ratio
                roof_ratio = torch.sum(label_patch)/prod(label_patch.shape)
                if roof_ratio < self.min_roof_ratio:
                    continue

                sample = {"image": image_patch, "label": label_patch}
                if self.transform:
                    sample = self.transform(sample)

                # if self.train:
                # For training, ignore patches without a minimum roof pixel ratio
                roof_ratio = torch.sum(sample["label"])/prod(sample["label"].shape)
                if roof_ratio < self.min_roof_ratio:
                    continue

                yield sample

    def __iter__(self):
        return chain.from_iterable(map(self.process_sample, cycle(self.names)))


class SingleImage(IterableDataset):
    def __init__(
        self,
        image_path: Path,
        patch_size: int = 512,
        patch_stride: int = 256,
    ):
        self.names = [image_path]
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def process_sample(self, name):
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

    def __iter__(self):
        return chain.from_iterable(map(self.process_sample, self.names))


class PILToTensor:
    def __init__(self):
        self.transform = transforms.PILToTensor()

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        return {"image": self.transform(image), "label": self.transform(label)}


class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() < self.p:
            image = hflip(image)
            label = hflip(label)

        return {"image": image, "label": label}


class RandomVerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() < self.p:
            image = vflip(image)
            label = vflip(label)

        return {"image": image, "label": label}


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size
        if not isinstance(self.output_size, list):
            self.output_size = [self.output_size, self.output_size]

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.output_size
        )
        image = crop(image, i, j, h, w)
        label = crop(label, i, j, h, w)

        return {"image": image, "label": label}


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees
        if not isinstance(self.degrees, list):
            self.degrees = [0, self.degrees]

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        angle = transforms.RandomRotation.get_params(degrees=self.degrees)
        image = rotate(image, angle)
        label = rotate(label, angle)

        return {"image": image, "label": label}


if __name__ == "__main__":
    dataset = AIRSDataset(
        Path(
            "/Users/jordan/Documents/Work/Job Search/Invision AI/Recruiting exercise ML/AIRS/val/image"
        ),
        Path(
            "/Users/jordan/Documents/Work/Job Search/Invision AI/Recruiting exercise ML/AIRS/val/label"
        ),
        Path(
            "/Users/jordan/Documents/Work/Job Search/Invision AI/Recruiting exercise ML/AIRS/val.txt"
        ),
        train=False
    )

    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size)

    debug_path = Path("/Users/jordan/Documents/Work/Job Search/Invision AI/Recruiting exercise ML/debug")

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
