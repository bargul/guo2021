from configuration import *
from voc_dataset import *
from ClassAwareSampler import *
from Network import *

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 

# Dataset
transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize([224,224])])
training_data = VOC(root=lt_dataset_output_path, imgtransform=transform_train)

train_count = int(len(training_data)*(1-validation_set_ratio))
val_count = len(training_data) - train_count
train_set, val_set = random_split(training_data, [train_count, val_count], generator=torch.Generator().manual_seed(seed))
train_set = train_set.dataset
val_set = val_set.dataset


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

uniform_dataloader = DataLoader(train_set, batch_size=batch_size_, shuffle=False)
rsampler = ClassAwareSampler(dataset=train_set,num_sample_class=len(labels_map),samples_per_gpu=batch_size_)
rebalanced_dataloader = DataLoader(train_set, batch_size=batch_size_, shuffle=False,sampler=rsampler)
optimizer = torch.optim.SGD(net.parameters(), weight_decay = weight_decay_, lr=lr_, momentum=momentum_)

uniform_dataloader_val = DataLoader(val_set, batch_size=batch_size_, shuffle=False)

Lcls = nn.BCEWithLogitsLoss()
Lcon = nn.MSELoss()

# Tensorboard
writer = SummaryWriter(tnsrbrd_dir)

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
    F1scr = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, F1scr

def calcLoss(uniform_dataloader, rebalanced_dataloader):
  xU, yU = next(iter(uniform_dataloader))
  xR, yR = next(iter(rebalanced_dataloader))

  loss = torch.tensor([0.0], device = dev)
  if uniform_branch_active:
    u , uHat = net(xU)
    loss += Lcls(u,yU)
  if resampled_branch_active:
    rHat , r = net(xR)
    loss += Lcls(r,yR)
  if uniform_branch_active and resampled_branch_active and logit_consistency:
    loss +=lambda_ * ( Lcon(u,uHat) + Lcon(r,rHat))
  return loss, u , r, uHat, rHat, xU, yU, xR, yR

total_iterations = int(len(train_set) / batch_size_)
total_iterations_val = int(len(val_set) / batch_size_)
patience = 0
epoch_counter = 1 
F1scr_val = 0
while patience < patience_level:
  avg_loss_iters = 0
  avg_loss_iters_val = 0
  # Train
  net.train()
  for i in tqdm(range(total_iterations), desc="train", colour='green'):
      optimizer.zero_grad()
      loss, u, r, uHat, rHat, xU, yU, xR, yR = calcLoss(uniform_dataloader, rebalanced_dataloader)
      avg_loss_iters += loss.item()
      loss.backward()
      optimizer.step()
  avg_loss_iters = avg_loss_iters / total_iterations
  writer.add_scalar("Training Set Loss ", avg_loss_iters, epoch_counter) 
  rR = (r>threshold).float()
  precision, recall, F1scr = F1_score(yR, rR)
  # Validation
  net.eval()
  for i in tqdm(range(total_iterations_val), desc="val", colour='blue'):
    loss, u_val, r_val, uHat_val, rHat_val, xU_val, yU_val, xR_val, yR_val = calcLoss(uniform_dataloader_val, uniform_dataloader_val)
    avg_loss_iters += loss.item()
  avg_loss_iters_val = avg_loss_iters_val / total_iterations_val
  writer.add_scalar("Validation set loss", avg_loss_iters_val, epoch_counter) 
  rR_val = (r_val>threshold).float()
  precision_val, recall_val, F1scr_val_new = F1_score(yR_val, rR_val)

  if F1scr_val_new > F1scr_val:
    F1scr_val = F1scr_val_new 
    modelsavename = "weights_{}.weights".format(epoch_counter)
    torch.save(net.state_dict(), modelsavename)
    print("saved",modelsavename)


print("End")
