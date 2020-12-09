#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/12/8 14:15
#@Author  :Wenbo 
#@FileName: data_pre.py

import os


all_file_dir = r'C:\Users\11982\Desktop\dataset\luomu_detect\trainset'

f = open(r'C:\Users\11982\Desktop\paddle\train.txt', 'w')

label_id = 0

class_list = [c for c in os.listdir(all_file_dir) if os.path.isdir(os.path.join(all_file_dir, c))]

# print(class_list)
for class_dir in class_list:


    image_path_pre = os.path.join(all_file_dir, class_dir)

    for img in os.listdir(image_path_pre):
        # print(img)
        f.write("{0}\t{1}\n".format(os.path.join(image_path_pre, img), label_id))

    label_id += 1

