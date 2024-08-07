import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    """ResNet Model modified for regression"""

    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept 1 channel (grayscale)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Modify the last fully connected layer for regression
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)
