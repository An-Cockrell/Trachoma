import torch
import copy
import torch.nn as nn
from torchvision import transforms, models


class resnet101(nn.Module):
    def __init__(self):
        super(resnet101, self).__init__()

        self.res101 = models.resnet101(pretrained=False)
        num_features = self.res101.fc.in_features
        self.res101.fc = nn.Linear(num_features, 1)

        # self.setup()

    def setup(self):
        self.avgpool = torch.nn.Sequential(*(list(self.res101.children())[-2:-1]))
        self.fc = torch.nn.Sequential((list(self.res101.children())[-1]))

        self.res101 = torch.nn.Sequential(*(list(self.res101.children())[:-2]))

        # del self.res101

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.res101(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # don't forget the pooling
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.fc(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.res101(x)