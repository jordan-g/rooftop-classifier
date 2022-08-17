import random

import torchvision.transforms as transforms
from torchvision.transforms.functional import center_crop, hflip, rotate, vflip


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


class CenterCrop:
    def __init__(self, output_size):
        self.output_size = output_size
        if not isinstance(self.output_size, list):
            self.output_size = [self.output_size, self.output_size]

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        image = center_crop(image,self.output_size)
        label = center_crop(label,self.output_size)

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