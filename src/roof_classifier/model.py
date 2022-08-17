import torch.nn as nn
from torchsummary import summary

from roof_classifier.utils import get_device

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (5, 5), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (2, 2)),
            nn.Conv2d(32, 16, (3, 3), padding=1),
            nn.ConvTranspose2d(16, 1, (3, 3), stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    device = get_device()
    
    model = Classifier()
    model.to(device)

    summary(model, (3, 512, 512))
