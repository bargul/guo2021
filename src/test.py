from configuration import *
from voc_dataset import *
from ClassAwareSampler import *
from Network import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Initials
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

# Dataset
transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize([224,224])])
test_data = VOC(root=test_dataset_output_path, imgtransform=transform_train)
testLoader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
net = torch.load(test_weight_path)
net = net.to(device)
net.eval()
  
for testData,testLabel in testLoader:
    u , r = net(testData)
    result = torch.nn.functional.relu((u+r)/2)
    print("test")


print("End")



