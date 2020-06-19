import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        features = []
        for i, module in self.submodule._modules.items():
            x = module(x)
            if int(i) > 3:
                features.append(x)
        return features


class Backbone(nn.Module):
    def __init__(self, network='resnet34', pretrained=True):
        super(Backbone, self).__init__()
        resnet = getattr(models, network)(pretrained=pretrained)
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        # this part is complicated, it is to get the number of channels of every feature_map
        self.feat_channels = []
        for i, module in backbone._modules.items():
            if int(i) > 3:
                self.feat_channels.append(list(list(module.children())[-1].children())[-1].num_features)

        self.extractor = FeatureExtractor(backbone)

    def forward(self, x):
        features = self.extractor(x)
        return features[::-1]


if __name__ == "__main__":
    backbone = Backbone()
    print(backbone.feat_channels)


