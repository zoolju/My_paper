# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:55:02 2020

@author: 1
"""

import numpy as np 

dict_     = np.load('A:/f-SGL-master/data/dic_2020_30.npy')
out_file  = open('A:/f-SGL-master/data/train_label_select.txt','w',encoding='utf-8')
with open('A:/f-SGL-master/data/label_2020_copy.txt','r') as f:
    for line in f:
        words = line.strip('\n').split()
        for word in words:
            if word in dict_:
                out_file.write(word+' ')
        out_file.write('\n')
out_file.close()

count = 1
flag  =0
with open('A:/f-SGL-master/data/train_label_select.txt','r') as f:
    for line in f:
        if line == '\n':
            print(count)
            flag += 1
        count += 1
    print('miss:,',flag)

