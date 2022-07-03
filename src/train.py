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
from copy import deepcopy
import time


# Dataset
transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize([224,224])])
training_data = VOC(root=lt_dataset_output_path, imgtransform=transform_train)

test_data = VOC(root=test_dataset_output_path, imgtransform=transform_train)
testLoader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

def splitDataset(training_data, validation_set_ratio):
  train_count = int(len(training_data)*(1-validation_set_ratio))
  val_count = len(training_data) - train_count
  train_set0, val_set0 = random_split(training_data, [train_count, val_count], generator=torch.Generator().manual_seed(seed))
  train_idxs = train_set0.indices
  train_set = deepcopy(training_data)
  train_set.reduceByIndexing(train_idxs)
  val_idxs = val_set0.indices
  val_set = deepcopy(training_data)
  val_set.reduceByIndexing(val_idxs)
  return train_set, val_set

train_set, val_set = splitDataset(training_data, validation_set_ratio)
if False: # quick test
  train_set, val_set = splitDataset(val_set, validation_set_ratio)
  
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
writer = SummaryWriter("{}/{}".format(tnsrbrd_dir,int(time.time())))

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
    #print("Acc:{} Precision:{} Recall:{}".format(accuracy,precision,recall))
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

def runOnDataset(dataloader,net,name,colour="blue"):
  iterations = int(len(dataloader) / batch_size_)
  counter = 0
  precision = 0
  recall = 0
  F1scr = 0
  net.eval()
  for i in tqdm(range(iterations), desc=name, colour=colour):
    counter+= 1
    x, y = next(iter(dataloader))
    uniform , resampled = net(x)

    inference_result = ( uniform + resampled )/ 2
    inference_result = (inference_result>threshold).float()

    tmp_precision, tmp_recall , tmp_f1 = F1_score(y, inference_result)
    precision += tmp_precision.item()
    recall += tmp_recall.item()
    F1scr += tmp_f1.item()
  precision /= counter
  recall /= counter
  F1scr /= counter
  return precision,recall,F1scr    



total_iterations = int(len(train_set) / batch_size_)
total_iterations_val = int(len(val_set) / batch_size_)
patience = 0
epoch_counter = 0 
batch_counter = 0
F1scr_val = 0
while patience < patience_level:
  epoch_counter += 1
  avg_loss_iters = 0
  ##########  Train #############
 
  for i in tqdm(range(total_iterations), desc="train", colour='green'):
      net.train()
      batch_counter += 1
      optimizer.zero_grad()
      loss, u, r, uHat, rHat, xU, yU, xR, yR = calcLoss(uniform_dataloader, rebalanced_dataloader)
      avg_loss_iters += loss.item()
      loss.backward()
      optimizer.step()
      writer.add_scalar("Training/loss", loss, batch_counter)

      uniform_result = (u>threshold).float()
      uniform_precision, uniform_recall, uniform_F1scr = F1_score(yU, uniform_result)

      resampled_result = (r>threshold).float()
      resampled_precision, resampled_recall, resampled_F1scr = F1_score(yR, resampled_result)

      writer.add_scalar("Training/uniform/precision", uniform_precision, batch_counter)
      writer.add_scalar("Training/uniform/recall", uniform_recall, batch_counter)
      writer.add_scalar("Training/uniform/F1scr", uniform_F1scr, batch_counter)

      writer.add_scalar("Training/resampled/precision", resampled_precision, batch_counter)
      writer.add_scalar("Training/resampled/recall", resampled_recall, batch_counter)
      writer.add_scalar("Training/resampled/F1scr", resampled_F1scr, batch_counter)

      if batch_counter % 20 == 0:
        ##########  Validation #############
        precision_val,recall_val,F1scr_val = runOnDataset(uniform_dataloader_val,net,"val")          
        writer.add_scalar("Validation/precision", precision_val, batch_counter)
        writer.add_scalar("Validation/recall", recall_val, batch_counter)
        writer.add_scalar("Validation/F1scr", F1scr_val, batch_counter)

        ##########  Test #############
        precision_test,recall_test,F1scr_test = runOnDataset(testLoader,net,"test","red")          
        writer.add_scalar("Test/precision", precision_test, batch_counter)
        writer.add_scalar("Test/recall", recall_test, batch_counter)
        writer.add_scalar("Test/F1scr", F1scr_test, batch_counter)


  avg_loss_iters = avg_loss_iters / total_iterations
  writer.add_scalar("Training/avg_loss", avg_loss_iters, epoch_counter)
   
  '''
  if F1scr_val_new > F1scr_val:
    F1scr_val = F1scr_val_new 
    modelsavename = "weights_{}.weights".format(epoch_counter)
    torch.save(net.state_dict(), modelsavename)
    print("saved",modelsavename)
  '''

print("End")
