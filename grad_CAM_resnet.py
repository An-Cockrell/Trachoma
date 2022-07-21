import torch
import copy
import torch.nn as nn
from torchvision import transforms, models

class resnet101(nn.Module):
    def __init__(self):
        super(resnet101, self).__init__()

        self.res101 = models.resnet101(pretrained=False)

        self.fc = copy.deepcopy(self.res101.fc)
        self.avgpool = copy.deepcopy(self.res101.avgpool)

        del self.res101.fc
        del self.res101.avgpool

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
        return self.features_conv(x)