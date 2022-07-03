from configuration import *
from voc_dataset import *
from ClassAwareSampler import *
from Network import *

import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

from tqdm import tqdm

# Initials
if torch.cuda.is_available():  
  devtype = "cuda:0" 
else:  
  devtype = "cpu"  
dev = torch.device(devtype)  

# Dataset
transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize([224,224])])
training_data = VOC(root=lt_dataset_output_path, imgtransform=transform_train)

if debug_mode:
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


net = Net().to(dev)
print(net)

uniform_dataloader = DataLoader(training_data, batch_size=batch_size_, shuffle=False)
rsampler = ClassAwareSampler(dataset=training_data,num_sample_class=len(labels_map),samples_per_gpu=batch_size_)
rebalanced_dataloader = DataLoader(training_data, batch_size=batch_size_, shuffle=False,sampler=rsampler)
optimizer = torch.optim.SGD(net.parameters(), weight_decay = weight_decay_, lr=lr_, momentum=momentum_)

Lcls = nn.BCEWithLogitsLoss()
Lcon = nn.MSELoss()

total_iterations = int(len(training_data) / batch_size_)

# https://discuss.pytorch.org/t/is-there-any-nice-pre-defined-function-to-calculate-precision-recall-and-f1-score-for-multi-class-multilabel-classification/103353
def F1_score(prob, label):
    prob = prob.bool()
    label = label.bool()
    epsilon = 1e-7
    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + epsilon))
    recall = torch.mean(TP / (TP + FN + epsilon))
    print("Acc:{} Precision:{} Recall:{}".format(accuracy,precision,recall))
    F2 = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, F2

threshold = torch.tensor([0.5]).to(dev)
net.train()
for i in tqdm(range(total_iterations), desc="i", colour='green'):
    print("total_iterations: {}, i: {}".format(total_iterations, i))
    optimizer.zero_grad()
    xR, yR = next(iter(rebalanced_dataloader))
    xU, yU = next(iter(uniform_dataloader))
    
    loss = torch.tensor([0.0], device = dev)
    if uniform_branch_active:
      u , uHat = net(xU)
      loss += Lcls(u,yU)
    if resampled_branch_active:
      rHat , r = net(xR)
      loss += Lcls(r,yR)
    if uniform_branch_active and resampled_branch_active and logit_consistency:
      loss +=lambda_ * ( Lcon(u,uHat) + Lcon(r,rHat))
    
    loss.backward()
    optimizer.step()
    
    rR = (r>threshold).float()*1
    F1_score(yR, rR)
    
    if i % 100 == save_weight_interval:
      
      name = "weights_{}.weights".format(i)
      torch.save(net, name)
      print("saved",name)


print("End")



