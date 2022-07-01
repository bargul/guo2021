import torch
from torchvision import models
import torch.nn as nn

debugMode = False

if debugMode:
    resnet50 = models.resnet50(pretrained=True)
    # .children(), .modules() difference: https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/3
    children = resnet50.children()
    layers = list(children)
    layer5 = list(layers[5].children())
    layer6 = list(layers[6].children())
    layer7 = list(layers[7].children())

class model1:
    def __init__(self):    
        super(self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # .children(), .modules() difference: https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/3
        layers = list(resnet50.children())
        self.backbone = nn.Sequential(*layers[:6])
        self.f1 = nn.Sequential(*layers[7])
        self.g1 = nn.Sequential(*layers[7])
        self.f2 = nn.Sequential(*layers[8:])
        self.g2 = nn.Sequential(*layers[8:])
    def forward(self, x, mode):
        x = self.backbone(x)
        if mode == "u":
            x = self.f1(x)
            x = self.f2(x)
        elif mode == "r":
            x = self.g1(x)
            x = self.g2(x)
        elif mode == "ur":
            



print("dummy1_1")
print("dummy1_2")
print("dummy1_3")