# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:04:56 2020

@author: 1
"""


import pandas as pd
import numpy as np
dictionary   = 'A:/f-SGL-master/data/dic_2021_.txt'
train_multi_labels = 'A:/f-SGL-master/data/train_label_select.txt'
#test_multi_labels = 'A:/f-SGL-master/data/train_label_select.txt'

# =============================================================================
# def co_matrix(dictionary,train_multi_labels):
# =============================================================================
result=[]
with open(dictionary,'r') as f:
	for line in f:
		result.append(line.strip('\n'))

list=result
for i in range(len(result)):
    result[i] = str(result[i])
word_list=[]
for i in range(0,len(list)):
    for j  in range(0,len(list)):
        word_list.append([list[i],list[j]])
 
cooc = np.zeros((len(list),len(list)))
with open(train_multi_labels,'r',encoding='utf-8',errors='ignore') as f:
    txt_list=f.read().strip().split("\n")
    labels = np.zeros((len(txt_list),len(list)))
    row = 0
    for line in txt_list:
        words = line.split()
        for ind_1 in range(len(words)):
            x         = list.index(words[ind_1])
            labels[row,x] = 1
            for ind_2 in range(len(words)):
                y         = list.index(words[ind_2])
                if(row<2250):
                    cooc[x,y] += 1
        row += 1
co_matrix = np.zeros((len(list),len(list)))
for i in range(cooc.shape[0]):
    for j in range(cooc.shape[1]):
        co_matrix[i,j] = cooc[i,j]/cooc[i,i]


np.save('A:/f-SGL-master/data/co_matrix.npy',co_matrix)
np.save('A:/f-SGL-master/data/train_labels.npy',labels[0:2250,:])
np.save('A:/f-SGL-master/data/val_labels.npy',labels[2250:2500,:])
np.save('A:/f-SGL-master/data/test_labels.npy',labels[2500:2750,:])
np.save('A:/f-SGL-master/data/appear_num.npy',cooc)



# =============================================================================
# if __name__=="__main__":
#     co_matrix(dictionary,train_multi_labels)
# =============================================================================
    
