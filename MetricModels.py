import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Resnet18(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet18, self).__init__()

        self.res18 = models.resnet18(pretrained=pretrained)
        num_features = self.res18.fc.in_features
        self.res18.fc = nn.Linear(num_features, 1)

        modules = list(self.res18.children())[:7]
        self.conv = nn.Sequential(*modules)

        modules = list(self.res18.children())[7:-1]
        self.classifier = nn.Sequential(*[*modules, Flatten(), list(self.res18.children())[-1]])

    def forward(self, x, full=True):
        x = self.conv(x)
        if not full:
            return torch.flatten(x, 1)
        else:
            classification = self.classifier(x)
            return classification


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x