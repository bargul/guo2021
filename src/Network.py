import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

debugMode = False

if debugMode:
    resnet50 = models.resnet50(pretrained=True)
    # .children(), .modules() difference: https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/3
    children = resnet50.children()
    layers = list(children)
    layer5 = list(layers[5].children())
    layer6 = list(layers[6].children())
    layer7 = list(layers[7].children())

class SubNet(nn.Module):
    def __init__(self,layers):    
        super(SubNet, self).__init__()
        self.lastStage = nn.Sequential(*layers[0])
        self.fullyConnected = nn.Sequential(*layers[1:]) 

    def forward(self,x):
        x = self.lastStage(x)
        x = self.fullyConnected(x)
        return x

class Net(nn.Module):
    def __init__(self):    
        super(Net,self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # .children(), .modules() difference: https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/3
        layers = list(resnet50.children())
        self.backbone = nn.Sequential(*layers[:6])
        self.subnetU = SubNet(layers[7:])
        self.subnetR = SubNet(layers[7:])

    def forward(self, x):
        x = self.backbone(x)
        u = self.subnetU.forward(x)
        r = self.subnetR.forward(x)
        return u,r    
            
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension (see below)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
print("dummy1_1")
print("dummy1_2")
print("dummy1_3")