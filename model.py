import torch.nn as nn
from backbones import Backbone
from sem_seg_heads import SemSegHead


class Net(nn.Module):
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class ResUnet(nn.Module):
    def __init__(self, network, pretrained=True, num_classes=3):
        super(ResUnet, self).__init__()
        self.backbone = Backbone(network=network, pretrained=pretrained)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = SemSegHead(self.backbone.feat_channels, num_classes=num_classes)
        self.net = Net(self.backbone, self.head)

    def forward(self, x):
        return self.net(x)




