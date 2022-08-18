# Rooftop Classifier
This repo contains a simple implementation of a semantic segmentation model trained to classify rooftops from aerial imagery using PyTorch. Training uses the freely-available [AIRS dataset](https://www.airs-dataset.com).

## Requirements
This package requires the following modules:

- setuptools>=61.0
- torch>=1.12
- torchvision>=0.13
- opencv-python>=4.6
- torchsummary>=1.5
- numpy>=1.23
- Pillow>=9.2
- PyYAML>=6.0

## Configuration
The paths to dataset files and inference image are set in `dataset.yaml`. By default, the AIRS dataset and test image are assumed to be within the root package directory, like so:

```
roof_classifier/
├─ src/
│  ├─ roof_classifier/
├─ AIRS/
│  ├─ train/
│  ├─ val/
│  ├─ train.txt
│  ├─ val.txt
├─ test_image.tif
```

## Usage
Install the package using the command:
```
pip install -e .
```

Then, run training and validation pipeline using:
```
cd src/roof_classifier && python train.py
```