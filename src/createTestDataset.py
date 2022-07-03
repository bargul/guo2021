import numpy as np
import random
import shutil

dataset_labels = {}

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


def copy_labelled_images(label,outputFolder):
    global dataset_labels
    
    train_path = "./dataset_org/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/{}_test.txt".format(label)
    count = 0
    with open(train_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(" ")
            if len(parts) == 3 and int(parts[-1])==1:
                count += 1
                image_name = parts[0]
                src_file = "./dataset_org/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format(image_name)
                print("{} - {}".format(image_name,label))
                shutil.copy(src_file,outputFolder)
                if image_name in dataset_labels.keys():
                    dataset_labels[image_name][voc_labels[label]] = 1
                else:
                    dataset_labels[image_name] = [0]*len(voc_labels.keys())
                    dataset_labels[image_name][voc_labels[label]] = 1

for label  in voc_labels.keys():
    copy_labelled_images(label,"./dataset_voc_test/images/") 

for image_name in dataset_labels.keys():
    label_path = "./dataset_voc_test/labels/{}.txt".format(image_name)
    with open(label_path,"w") as fp:
        for elem in dataset_labels[image_name]:
            fp.write("{}\n".format(elem))
