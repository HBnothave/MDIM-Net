import torch
import torch.nn as nn

class MultiStageClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(3):
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, current_channels * 2, 3, padding=1),
                    nn.BatchNorm2d(current_channels * 2),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )
            current_channels *= 2

        self.up_layers = nn.ModuleList()
        for _ in range(3):
            self.up_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current_channels, current_channels // 2, 3, padding=1),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU()
                )
            )
            current_channels = current_channels // 2

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes)
        )

    def forward(self, x):
        skips = []
        for layer in self.down_layers:
            x = layer(x)
            skips.append(x)

        for layer, skip in zip(self.up_layers, reversed(skips)):
            x = layer(x + skip)

        return self.final(x)