# from dataloader.trsfrms import must_transform
from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

# reference: https://github.com/andrewssobral/deep-learning-pytorch/blob/master/segmentation/utils/dataset.py

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def get_full_path(root, basename, extension):
    return os.path.join(root, '{basename}{extension}'.format(**locals()))

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

# Standard Pascal VOC format
class VOC(Dataset):

    def __init__(self, root, imgtransform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')
        
        self.filenames = [image_basename(f)
            for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames.sort()

        self.imgtransform = imgtransform
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"  
        self.device = torch.device(dev)  


    def __getitem__(self, index):
        filename = self.filenames[index]

        image = np.array(Image.open(get_full_path(self.images_root, filename, '.jpg')).convert('RGB'))

        with open(get_full_path(self.labels_root, filename, '.txt')) as f:
            lines = f.readlines()
            label = [int(currentline[0]) for currentline in lines]
        if self.imgtransform is not None:
            image = self.imgtransform(image)
            image = image.to(self.device)

        return image, label

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    img_mean = np.array([104, 117, 128]).reshape(1, 1, 3)
    transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize([224,224])])
    myvoc = VOC(root="C:/aal/lrn/METU/CENG502/proj/r1/guo2021/dataset_voc_lt", imgtransform = transform_train)
    img, lbl = myvoc[1]
    plt.imshow(img)
    img = Image.fromarray(img)
    img.show()