import torch
import torch.nn as nn
import torchvision
import os


class myresnet(nn.Module):
    def __init__(self):
        super(myresnet, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, 12)

    def forward(self, X):
        x = self.backbone(X)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


