import torch
from torch import nn


class StupidNet(nn.Module):
    def __init__(self, width, height):
        super(StupidNet, self).__init__()

        self.flatten = nn.Flatten()
        self.linearStack = nn.Sequential(
            nn.Linear(width * height, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearStack(x)
        return logits


class MidNet(nn.Module):
    def __init__(self):
        super(MidNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output
