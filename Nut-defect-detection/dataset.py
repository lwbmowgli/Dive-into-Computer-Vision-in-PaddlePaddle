#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/12/8 13:52
#@Author  :Wenbo 
#@FileName: dataset.py.py

import numpy as np
from PIL import Image
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle as pd

class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, txt, transform=None):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()
        imgs = []
        f = open(txt, 'r')
        for line in f:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            # self.loader = loader
    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]
        # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn)
        img = img.convert("RGB")

        img =  np.array(img).astype('float32')
        img *= 0.007843
        label = np.array([label]).astype(dtype='int64')
        # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)
            # 数据标签转换为Tensor
        return img, label
        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


# if __name__ == '__main__':
#     transform = T.Compose([
#                         T.RandomResizedCrop([448,448]),
#                         T.Transpose(),
#                         T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#                     ])
#
#     train_data=MyDataset(txt=r'C:\Users\11982\Desktop\paddle\train.txt', transform=transform)
#
#     for i in range(len(train_data)):
#         sample = train_data[i]
#         print(sample[0], sample[1].shape)
