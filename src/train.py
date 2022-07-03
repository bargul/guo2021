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

# Initials
if torch.cuda.is_available():  
  devtype = "cuda:0" 
else:  
  devtype = "cpu"  
dev = torch.device(devtype)  

# Dataset
transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize([224,224])])
training_data = VOC(root=lt_dataset_output_path, imgtransform=transform_train)

train_count = int(len(training_data)*(1-validation_set_ratio))
val_count = len(training_data) - train_count
train_set, val_set = random_split(training_data, [train_count, val_count], generator=torch.Generator().manual_seed(seed))


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

# Tensorboard
writer = SummaryWriter(tnsrbrd_dir)


total_iterations = int(len(training_data) / batch_size_)
patience = 0
epoch_counter = 1 
while patience < patience_level:
  avg_loss_iters = 0
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
      avg_loss_iters += loss.item()
      loss.backward()
      optimizer.step()

  avg_loss_iters = avg_loss_iters / total_iterations
  writer.add_scalar("Training/avg_loss", avg_loss_iters, epoch_counter) 
  net.eval()

  if i % 100 == save_weight_interval:
    name = "weights_{}.weights".format(i)
    torch.save(net, name)
    print("saved",name)


print("End")



