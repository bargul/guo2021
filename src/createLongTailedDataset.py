import numpy as np
import random
import shutil

voc_labels = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
    ]


def count_image_number(label):
    train_path = "../dataset_org/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/{}_train.txt".format(label)
    count = 0
    with open(train_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(" ")
            if len(parts) == 3 and int(parts[-1])==1:
                count += 1
    print("{} has {} number of labelled image".format(label,count))
    return count

def copy_number_of_labelled_image(label,number,outputFolder):
    train_path = "../dataset_org/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/{}_train.txt".format(label)
    count = 0
    with open(train_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(" ")
            if len(parts) == 3 and int(parts[-1])==1:
                count += 1
                src_file = "../dataset_org/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/{}.jpg".format(parts[0])
                shutil.copy(src_file,outputFolder)
                if count == number:
                    return

# number of images per class ranges from 4 to 775
# The 20 classes are split into three groups according to the
# number of training samples per class: a head class has more
# than 100 samples, a medium class has 20 to 100 samples,
# and a tail class has less than 20 samples. The ratio of head,
# medium and tail classes after such splitting is 6:6:8

head = np.random.randint(low=100,high=775, size=6)
medium = np.random.randint(low=20,high=100, size=6) 
tail = np.random.randint(low=4,high=20, size=8) 
class_distribution = np.concatenate((head,medium,tail),axis=0)
class_distribution = np.sort(class_distribution)[::-1]

labels = []
for label in voc_labels:
   labels.append((label,count_image_number(label)))

labels_sorted = sorted(labels, key=lambda tup: tup[1],reverse=True)

for label_tuple , target_image_number in zip(labels_sorted,class_distribution.tolist()):
    print(label_tuple,target_image_number)
    copy_number_of_labelled_image(label_tuple[0],target_image_number,"../dataset_voc_lt/images/")
    