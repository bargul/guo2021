from __future__ import division
# import math
import torch
import numpy as np
# from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from torch.utils.data import DistributedSampler as _DistributedSampler
import random

# https://github.com/wutong16/DistributionBalancedLoss/blob/3ba665db6d2c1081d74f258b69ad39d58fba218f/mllt/datasets/loader/sampler.py#L294


class RandomCycleIter:

    def __init__(self, data_list, test_mode=False):
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=3, reduce = 4):
        random.seed(0)
        torch.manual_seed(0)
        num_classes = len(np.unique(data_source.CLASSES))

        self.epoch = 0

        self.class_iter = RandomCycleIter(range(num_classes))
        # cls_data_list = [list() for _ in range(num_classes)]
        '''
        labels = [ i for i in range(num_classes)]
        for i, label in enumerate(labels):
            cls_data_list[label].append(i)'''
        self.cls_data_list, self.gt_labels = data_source.get_index_dic(list=True, get_labels=True)

        self.num_classes = len(self.cls_data_list)
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list] # repeated
        self.num_samples = int(max([len(x) for x in self.cls_data_list]) * len(self.cls_data_list)/ reduce) # attention, ~ 1500(person) * 80
        self.num_samples_cls = num_samples_cls
        print('>>> Class Aware Sampler Built! Class number: {}, reduce {}'.format(num_classes, reduce))

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples

    def set_epoch(self,  epoch):
        self.epoch = epoch

    def get_sample_per_class(self):
        condition_prob = np.zeros([self.num_classes, self.num_classes])
        sample_per_cls = np.asarray([len(x) for x in self.gt_labels])
        rank_idx = np.argsort(-sample_per_cls)

        for i, cls_labels in enumerate(self.gt_labels):
            num = len(cls_labels)
            condition_prob[i] = np.sum(np.asarray(cls_labels), axis=0) / num

        sum_prob = np.sum(condition_prob, axis=0)
        need_sample = sample_per_cls / sum_prob
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.bar(range(self.num_classes), sum_prob[rank_idx], alpha = 0.5, color='green', label='sum_j( p(i|j) )')
        plt.legend()
        plt.hlines(1, 0, self.num_classes, linestyles='dashed', color='r', linewidth=1)
        ax2 = fig.add_subplot(2,1,2)
        # ax2.bar(range(self.num_classes), need_sample[rank_idx], alpha = 0.5, label='need_avg')
        ax2.bar(range(self.num_classes), sample_per_cls[rank_idx], alpha = 0.5, label='ori_distribution')
        plt.legend()
        plt.savefig('./coco_resample_deduce.jpg')
        print('saved at ./coco_resample_deduce.jpg')
        print(np.min(sum_prob), np.max(need_sample))
        exit()
