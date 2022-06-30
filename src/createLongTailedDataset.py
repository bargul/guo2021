# python3 createLongTailedDataset.py > ../dataset_voc_lt/info.txt
import numpy as np
import random
import shutil

long_tailed_dataset_labels = {}
long_tailed_dataset_info = []

voc_labels = {
    "aeroplane":0,
    "bicycle":1,
    "bird":2,
    "boat":3,
    "bottle":4,
    "bus":5,
    "car":6,
    "cat":7,
    "chair":8,
    "cow":9,
    "diningtable":10,
    "dog":11,
    "horse":12,
    "motorbike":13,
    "person":14,
    "pottedplant":15,
    "sheep":16,
    "sofa":17,
    "train":18,
    "tvmonitor":19
}


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
    global long_tailed_dataset_labels
    train_path = "../dataset_org/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/{}_train.txt".format(label)
    count = 0
    with open(train_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(" ")
            if len(parts) == 3 and int(parts[-1])==1:
                count += 1
                image_name = parts[0]
                src_file = "../dataset_org/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/{}.jpg".format(image_name)
                print("{} - {}".format(image_name,label))
                shutil.copy(src_file,outputFolder)
                if image_name in long_tailed_dataset_labels.keys():
                    long_tailed_dataset_labels[image_name][voc_labels[label]] = 1
                else:
                    long_tailed_dataset_labels[image_name] = [0]*len(voc_labels.keys())
                    long_tailed_dataset_labels[image_name][voc_labels[label]] = 1


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
for label in voc_labels.keys():
   labels.append((label,count_image_number(label)))

labels_sorted = sorted(labels, key=lambda tup: tup[1],reverse=True)

for label_tuple , target_image_number in zip(labels_sorted,class_distribution.tolist()):
    long_tailed_dataset_info.append((label_tuple,target_image_number))
    copy_number_of_labelled_image(label_tuple[0],target_image_number,"../dataset_voc_lt/images/") 

for elem in long_tailed_dataset_info:
    print(elem)

for image_name in long_tailed_dataset_labels.keys():
    label_path = "../dataset_voc_lt/labels/{}.txt".format(image_name)
    with open(label_path,"w") as fp:
        for elem in long_tailed_dataset_labels[image_name]:
            fp.write("{}\n".format(elem))
