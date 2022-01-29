# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:08:32 2020

@author: 1
"""
import numpy as np

s = []
in_file   = open('A:/f-SGL-master/data/label_2020.txt','r') #由于我使用的pycharm已经设置完了路径，因此我直接写了文件名
out_file  = open('A:/f-SGL-master/data/label_2020_30.txt','w',encoding='utf-8')
out_file1 = open('A:/f-SGL-master/data/label_2020_30_dict.txt','w',encoding='utf-8')
count = 1
check = []
for lines in in_file:
    # query_list.append(line.replace('/','').replace('、','').replace(' ','').strip('\n'))
    for word in lines.split():
        #if count == 1 or count == 4 or count == 5 or count == 6 or count == 7:
        if word not in check:
            check.append(word)
            out_file.write(word+' ')
            out_file1.write(word+'\n')
            count += 1
    out_file.write('\n')
in_file.close()
out_file.close() 
out_file.close()


def word_count(file_name):
    import collections
    word_freq = collections.defaultdict(int)
    with open(file_name) as f:
        for l in f:
            for w in l.strip().split():  
                word_freq[w] += 1
    return word_freq

def build_dict(file_name, min_word_freq=40):
    word_freq = word_count(file_name) 
    a = word_freq
    word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items()) 
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, range(len(words))))
    word_idx['<unk>'] = len(words) #unk表示unknown，未知单词
    return word_idx,a

dictionary,word_freq = build_dict('A:/f-SGL-master/data/label_2020.txt',)

dic = []
for key in dictionary:
    dic.append(key)
dic.pop()
np.save('A:/f-SGL-master/data/dic_2020_30.npy',dic)
out_file  = open('A:/f-SGL-master/data/dic_2021_.txt','w',encoding='utf-8')
for i in range(len(dic)):
    out_file.write(dic[i]+'\n')
out_file.close()
    