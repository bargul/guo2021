# from dataloader.trsfrms import must_transform
from fileinput import filename
from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder
#from torchvision import transforms as trs
from PIL import Image
import numpy as np
import os

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

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor')

    def __init__(self, root, imgtransform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')
        
        self.filenames = [image_basename(f)
            for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames.sort()

        self.imgtransform = imgtransform

    def __getitem__(self, index):
        filename = self.filenames[index]

        image = np.array(Image.open(get_full_path(self.images_root, filename, '.jpg')).convert('RGB'))

        # with open(get_full_path(self.images_root, filename, '.jpg'), 'rb') as f:
        #    image = load_image(f).convert('RGB')
        with open(get_full_path(self.labels_root, filename, '.txt')) as f:
            lines = f.readlines()
            label = [int(currentline[0]) for currentline in lines]
        if self.imgtransform is not None:
            image = self.imgtransform(image)

        return image, label

    def __len__(self):
        return len(self.filenames)

    def get_cat2imgs(self):
            """Get a dict with class as key and img_ids as values, which will be
            used in :class:`ClassAwareSampler`.
            Returns:
                dict[list]: A dict of per-label image list,
                the item of the dict indicates a label index,
                corresponds to the image index that contains the label.
            """
            if self.CLASSES is None:
                raise ValueError('self.CLASSES can not be None')
            # sort the label index
            cat2imgs = {i: [] for i in range(len(self.CLASSES))}
            for i in range(len(self)):
                cat_ids = set(self.get_cat_ids(i))
                for cat in cat_ids:
                    cat2imgs[cat].append(i)
            return cat2imgs

    def get_cat_ids(self, idx):
        """Get category ids by index.
        Args:
            idx (int): Index of data.
        Returns:
            list[int]: All categories in the image of specified index.
        """
        resultList = []
        filePath = "../dataset_voc_lt/labels/{}.txt".format(self.filenames[idx])
        with open(filePath,"r") as f:
            lines = f.readlines()
            for index,elem in enumerate(lines):
                elem = elem.strip()
                elem = int(elem)
                if elem == 1:
                    resultList.append(index)
        return resultList


if __name__ == '__main__':
    img_mean = np.array([104, 117, 128]).reshape(1, 1, 3)
    myvoc = VOC(root="C:/aal/lrn/METU/CENG502/proj/r1/guo2021/dataset_voc_lt")
    img, lbl = myvoc[1]
    plt.imshow(img)
    img = Image.fromarray(img)
    img.show()