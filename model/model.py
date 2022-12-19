import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

"""
models including efficientnet ,densenet_201, Resnext
"""


class efficientnet(nn.Module):
    def __init__(self):
        super(efficientnet, self).__init__()
        self.model = models.efficientnet_b4(pretrained=True)
        self.model.classifier[1] = nn.Linear(1792+128, 33, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.location = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            MyLinear(128),
        )

    def forward(self, input, location):
        if location is not None:
            location = location[:, :2]  # no time
            out = self.model.features(input)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out_location = self.location(location)
            agg_out = torch.cat([out, out_location], 1)
            output = self.model.classifier(agg_out)
        else:
            raise RuntimeError('Failed to open location database')

        return output


class densenet_201(nn.Module):
    def __init__(self):
        super(densenet_201, self).__init__()
        self.model = models.densenet201(pretrained=True)
        in_features = self.model.classifier.in_features + 128
        self.location = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            MyLinear(128),
        )
        self.model.classifier = nn.Linear(in_features=in_features, out_features=33, bias=True)

    def forward(self, input, location):
        if location is not None:
            location = location[:, :2] # no time
            out = self.model.features(input)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_location = self.location(location)
            agg_out = torch.cat([out, out_location], 1)
            output = self.model.classifier(agg_out)
        else:
            raise RuntimeError('Failed to open location database')
        return output

class MyLinear(nn.Module):
    """
    normalization layer for Resnest & Resnext
    """
    def __init__(self, linear_size, ):
        super(MyLinear, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.norm_fn1 = nn.LayerNorm(self.l_size)
        self.norm_fn2 = nn.LayerNorm(self.l_size)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.norm_fn1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        y = self.norm_fn2(y)
        out = x + y
        return out


class Resnext(nn.Module):
    def __init__(self):
        super(Resnext, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        in_features = 2048 + 128
        self.location = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            MyLinear(128),
        )
        self.model.fc = nn.Linear(in_features=in_features, out_features=33, bias=True)


    def forward(self, input, location):
        if location is not None:
            location = location[:, :2]
            out = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(input))))
            out = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(out))))
            out = self.model.avgpool(out)
            out = torch.flatten(out, 1)
            out_location = self.location(location)
            out = torch.cat([out, out_location], 1)
            out = self.model.fc(out)
        else:
            raise RuntimeError('Failed to open location database')
        return out
