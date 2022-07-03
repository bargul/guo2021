from voc_dataset import *
from ClassAwareSampler import *
from Network import *

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
#import matplotlib.pyplot as plt

debugMode = False

# Initials
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

# Dataset
training_data = VOC(root="../dataset_voc_lt")

labels_map = {
    0:"aeroplane",
    1:"bicycle",
    2:"bird",
    3:"boat",
    4:"bottle",
    5:"bus",
    6:"car",
    7:"cat",
    8:"chair",
    9:"cow",
    10:"diningtable",
    11:"dog",
    12:"horse",
    13:"motorbike",
    14:"person",
    15:"pottedplant",
    16:"sheep",
    17:"sofa",
    18:"train",
    19:"tvmonitor"
}

if debugMode:
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        label = label.index(1)
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

epoch = 10
batch_size_ = 64
lr_ = 0.01
momentum_ = 0.9
weight_decay_ = 0.0001
lambda_ = 0.1

net = Net().to(device)
print(net)


uniform_dataloader = DataLoader(training_data, batch_size=batch_size_, shuffle=False)
rsampler = ClassAwareSampler(dataset=training_data,num_sample_class=20,samples_per_gpu=batch_size_)
rebalanced_dataloader = DataLoader(training_data, batch_size=batch_size_, shuffle=False,sampler=rsampler)
optimizer = torch.optim.SGD(net.parameters(), weight_decay = weight_decay_, lr=lr_, momentum=momentum_)

Lcls = nn.BCEWithLogitsLoss()
Lcon = nn.MSELoss()

net.train()
for i in range(epoch):
    optimizer.zero_grad()
    xR, yR = next(iter(rebalanced_dataloader))
    xU, yU = next(iter(uniform_dataloader))
    print("dummy")

    '''
    u , uHat = net(xU)
    rHat , r = net(xR)
    loss = Lcls(u,yU) + Lcls(r,yR) + lambda_ * ( Lcon(u,uHat) + Lcon(r,rHat))
    loss.backward()
    optimizer.step()
    '''

print("End")



