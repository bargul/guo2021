# Long-Tailed Multi-Label Visual Recognition by Collaborative Training on Uniform and Re-balanced Samplings

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper : Long-Tailed Multi-Label Visual Recognition by Collaborative Training on
Uniform and Re-balanced Samplings 

The paper is published in CVPR2021 and achieves state of the art results. On top of achieving state of the art results they actually develop a novel method. Our goal is to reproduce authors' results in the same way they did. They simply and judiciously add one of their novelties at a time and show its indiviual contribution. So our goal is to reproduce the table in img4 by adding one method at a time. Even if we can't get numerically close results to the paper's we would like to similar effects of each proposed method's addition. 

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.



# 2. The method and my interpretation

## 2.1. The original method

- Creating long tailed dataset from Pascal Voc and Ms Coco with Pareto distribution.
- Creating a Network, the shared bottom network is the conventional ResNet50 excluding the last stage. 
- For Subnet-U and Subnet-R, author states that first include an identical copy of the last stage of ResNet,After that, a linear classifier in the form of a fully connected layer is added to each branch.
- Conventional Classification Loss, Logit Compensation and Logit Consistency between Branches is used as loss function.
- Run inference for each input test image, the predictions of two branches are averaged as the final prediction result.

## 2.2. Our interpretation 

- Creating Long tailed dataset part is only explained with a sentence Pareto distribution used, we follow the references but could not found the exact way so we use the 6-6-8 split which explained in papers with 4-20 , 20-100 and 100-775 image sample intervals. 
- We use Resnet50 Imagenet Pretrained model from torchvision model zoo, we are not sure that if its the same pretrained values. We split the network from last stage and try to create 2 subnet model from the excluding stage , to produce 20 class 1-hot vector we added linear layer with sigmoid activation layer.
- We used nn.BCEWithLogitsLoss() for Conventional Classification Loss and nn.MSELoss() for Logit Consistency. It is not explained clearly that how the uniform and resampled data will feed the system. What is the order ? The subnet interaction between each other were unclear. The interaction at the subnet among the net input from backbone, the and the output were unclear.
- For testing only one of the subnets (i.e Subnet-U) was only the respective subnet trained or was all the system trained then only the respective subnet used? We trained only respective subnet.
- Some parameters werw given but many were missing such as batch size, training epochs. We did not know if the mean substraction and normalized were applied? 
- The usage of validation set was unclear. We don't know how they used it , we split our training set as %90-%10.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.\
Original paper used Pytorch 1.4.0 version but we used 1.12.0 .\
We tried to follow the same procedure while preparing Long Tailed dataset but our set is different inevitably.

## 3.2. Running the code

You can find our version of Long tailed Voc Dataset with these links. \
[Pascal VOC]([http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)) \
[dataset_voc_test.zip]([https://drive.google.com/file/d/14zlzl8V-gI7f9hAlLgcZgoYSK9TJdBD7/view?usp=sharing](https://drive.google.com/file/d/14zlzl8V-gI7f9hAlLgcZgoYSK9TJdBD7/view?usp=sharing)) \
[dataset_voc_lt.zip]([https://drive.google.com/file/d/135dGh0ti0vvIbWeePVN66UQvkX8DKJAr/view?usp=sharing](https://drive.google.com/file/d/135dGh0ti0vvIbWeePVN66UQvkX8DKJAr/view?usp=sharing)) 

We used docker container with Pytorch 1.12.0 , Python 3.7.13 on Ubuntu 18.04 Desktop Pc. You can pull the latest docker image with these commands. \
`docker pull pytorch/pytorch:latest`  
`docker run --gpus all -v /guo2021:/guo2021 -w /guo2021 -it pytorch/pytorch:latest`

You can directly use our dataset.zip files by extracting them to dataset_voc_lt and dataset_voc_test or you can create your own version with \
`python src/createLongTailedDataset.py` \
`python src/createTestDataset.py`\
All the parameters can be found in configuration.py file. After setting paremeters properly you can directly run train.py file to start training. \
`python src/train.py` This code will store weight files in src/weight folder, you can use them later for testing. \
`python src/test.py` This code will run the test set.


├── dataset_org\
│   ├── VOCtest_06-Nov-2007.tar\
│   ├── VOCtest_06-Nov-2007\
│   └── VOCtrainval_11-May-2012.tar\
│   ├── VOCtrainval_11-May-2012\
├── dataset_voc_lt\
│   ├── dataset_voc_lt.zip\
│   ├── images\
│   └── labels\
│   ├── info.txt\
├── dataset_voc_test\
│   ├── dataset_voc_test.zip\
│   ├── images\
│   └── labels\
│   ├── info.txt\
├── src\
│   ├── ClassAwareSampler.py\
│   ├── configuration.py\
│   ├── createLongTailedDataset.py\
│   ├── createTestDataset.py\
│   ├── mean_ap.py\
│   ├── Network.py\
│   ├── test.py\
│   ├── train.py\
│   ├── VocDataset.py\
│   └── weights\
├── runs\
│   ├── 1656888576_only_uniform\
│   ├── 1656888901_only_rebalanced\
│   ├── 1656889605\
│   └── 1656889821\
├── dockerCommands.txt\
├── README.md\

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References
- Network.py - https://deci.ai/blog/measure-inference-time-deep-neural-networks/
- ClassAwareSampler.py - https://github.com/open-mmlab/mmdetection 
- VocDataset.py - https://github.com/andrewssobral/deep-learning-pytorch/blob/master/segmentation/utils/dataset.py
- train.py - https://discuss.pytorch.org/t/is-there-any-nice-pre-defined-function-to-calculate-precision-recall-and-f1-score-for-multi-class-multilabel-classification/103353

# Contact

- Baran Gülmez \
baran.gulmez07@gmail.com
- Ibrahim Koc \
iibrahimkoc@gmail.com
