# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:15:24 2020

@author: 1
"""

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import torch
import numpy as np

# 已有的glove词向量
dictionary   = 'A:/f-SGL-master/data/dic_2021_.txt'
result=[]
with open(dictionary,'r') as f:
	for line in f:
		result.append(line.strip('\n'))
test_sentence = result
vocab = set(test_sentence) # 通过set将重复的单词去掉
word_to_idx = {word: i for i, word in enumerate(vocab)}
# 定义了一个unknown的词，也就是说没有出现在训练集里的词，我们都叫做unknown，词向量就定义为0。

idx_to_word = {i: word for i, word in enumerate(vocab)}


# =============================================================================
# glove_file = datapath('B:/f-SGL-master/data/glove/glove.6B.300d.txt')
# # 指定转化为word2vec格式后文件的位置
# tmp_file = get_tmpfile('B:/f-SGL-master/data/glove/test_word2vec.txt')
# glove2word2vec(glove_file, tmp_file)
# =============================================================================

wvmodel = gensim.models.KeyedVectors.load_word2vec_format('A:/f-SGL-master/data/glove/test_word2vec.txt', binary=False, encoding='utf-8')
vocab_size = len(vocab)
embed_size = 300
weight = np.zeros((vocab_size, embed_size))
vec    = np.zeros((vocab_size, embed_size))
vec_label = []
flag = 1
for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
    #print(i,':done')
    #print(idx_to_word[index],weight[index, :], flag)
    flag+=1
    vec_label.append(idx_to_word[index])
    vec[result.index(idx_to_word[index]),:] = weight[index, :]
np.save('A:/f-SGL-master/data/Glove_vec.npy',vec)
np.save('A:/f-SGL-master/data/Dictionary_label.npy',vec_label)

