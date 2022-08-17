import random

import torchvision.transforms as transforms
from torchvision.transforms.functional import crop, hflip, rotate, vflip


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