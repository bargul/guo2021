# Long-Tailed Multi-Label Visual Recognition by Collaborative Training on Uniform and Re-balanced Samplings

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper : Long-Tailed Multi-Label Visual Recognition by Collaborative Training on
Uniform and Re-balanced Samplings 

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it. \
We used 

You can find our version of Long tailed Voc Dataset with these links. \
[Pascal VOC]([http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)) \
[dataset_voc_test.zip]([https://drive.google.com/file/d/14zlzl8V-gI7f9hAlLgcZgoYSK9TJdBD7/view?usp=sharing](https://drive.google.com/file/d/14zlzl8V-gI7f9hAlLgcZgoYSK9TJdBD7/view?usp=sharing)) \
[dataset_voc_lt.zip]([https://drive.google.com/file/d/135dGh0ti0vvIbWeePVN66UQvkX8DKJAr/view?usp=sharing](https://drive.google.com/file/d/135dGh0ti0vvIbWeePVN66UQvkX8DKJAr/view?usp=sharing)) 

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
