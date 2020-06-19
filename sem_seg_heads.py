import torch.nn as nn
import torch
from backbones import Backbone


class SemSegHead(nn.Module):
    def __init__(self, feat_channels, num_classes=3):
        super(SemSegHead, self).__init__()
        self.num_classes = num_classes

        self.last_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.num_classes, 3, padding=1))

        self.feature_up = nn.ModuleList([])
        for i in range(len(feat_channels)):
            if i < len(feat_channels) - 1:
                self.feature_up.append(self.up_block(feat_channels[::-1][i], feat_channels[::-1][i] / 2))
            else:
                self.feature_up.append(self.up_block(feat_channels[::-1][i], feat_channels[::-1][i]))

    def forward(self, x):
        feature_sum = x[0]
        for i, module in enumerate(self.feature_up):
            if i < len(x) - 1:
                feature_sum = module(x[i]) + x[i+1]
            else:
                feature_sum = module(x[i])

        output = self.last_block(feature_sum)

        return torch.sigmoid(output)

    def up_block(self, in_channels, out_channels):
        out_channels = int(out_channels)
        # bottleneck architecture
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),

            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        return self.block


if __name__ == "__main__":
    backbone = Backbone()
    head = SemSegHead(backbone.feat_channels)
    print(head)


