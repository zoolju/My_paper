import numpy as np
import os
import random

img_dir = 'F:/multi_label/data/data/'
img_list  =[]
label_list=[]
sub_name = ["s_bailin","s_huangwei","s_lianlian","s_xiangchen","s_zhengzifeng"]
ROI_name = ["V1","V2","V3","LVC","OCC","HVC"]
count = 0
for sub in range(5):  
    for roi in range(3,6):
        img_dir = 'F:/multi_label/data/'+sub_name[sub]
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        file_dir = img_dir+'/'+ROI_name[roi]
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        train_data = np.load('F:\\multi_label\\data'+'\\category'+"\\"+sub_name[sub]+"\\"+ROI_name[roi]+'\\'+'Train2048_data_r.npy')
        test_data  = np.load('F:\\multi_label\\data'+'\\category'+"\\"+sub_name[sub]+"\\"+ROI_name[roi]+'\\'+'Test2048_data_r.npy')
        val_data = np.load('F:\\multi_label\\data'+'\\category'+"\\"+sub_name[sub]+"\\"+ROI_name[roi]+'\\'+'Val2048_data_r.npy')

        
        train_label = np.load('A:/f-SGL-master/data/train_labels.npy').T
        test_label  = np.load('A:/f-SGL-master/data/test_labels.npy').T
        val_label   = np.load('A:/f-SGL-master/data/val_labels.npy').T
        if count == 0:
            file1 = open('F:/multi_label/data/train_data_list.txt', 'w')
            file2 = open('F:/multi_label/data/train_label_list.txt', 'w')
            file3 = open('F:/multi_label/data/test_data_list.txt', 'w')
            file4 = open('F:/multi_label/data/test_label_list.txt', 'w')
            file5 = open('F:/multi_label/data/val_data_list.txt', 'w')
            file6 = open('F:/multi_label/data/val_label_list.txt', 'w')
        for i in range(2250):
            a = train_data[:,:,i]
            b = train_label[:,i]
            np.save(os.path.join(file_dir+'/'+'train_data{0:04d}').format(i),a)
            np.save(os.path.join(file_dir+'/'+'train_label{0:04d}').format(i),b)
            if count == 0:
                data = 'train_data{0:04d}.npy'.format(i)+'\n'
                file1.write(data)
                label = 'train_label{0:04d}.npy'.format(i)+'\n'
                file2.write(label)
        if count == 0:
            file1.close()
            file2.close()

        for i in range(250):
            a = test_data[:,:,i]
            b = test_label[:,i]
            np.save(os.path.join(file_dir+'/'+'test_data{0:04d}').format(i),a)
            np.save(os.path.join(file_dir+'/'+'test_label{0:04d}').format(i),b)
            if count == 0:
                data = 'test_data{0:04d}.npy'.format(i)+'\n'
                file3.write(data)
                label = 'test_label{0:04d}.npy'.format(i)+'\n'
                file4.write(label)
        for i in range(250):
            if count == 0:
                data = 'val_data{0:04d}.npy'.format(i)+'\n'
                file3.write(data)
                label = 'val_label{0:04d}.npy'.format(i)+'\n'
                file4.write(label)
        if count == 0:
            file3.close()
            file4.close()
        
        for i in range(250):
            a = val_data[:,:,i]
            b = val_label[:,i]
            np.save(os.path.join(file_dir+'/'+'val_data{0:04d}').format(i),a)
            np.save(os.path.join(file_dir+'/'+'val_label{0:04d}').format(i),b)
            if count == 0:
                data = 'val_data{0:04d}.npy'.format(i)+'\n'
                file5.write(data)
                label = 'val_label{0:04d}.npy'.format(i)+'\n'
                file6.write(label) 
        if count == 0:
            file5.close()
            file6.close()
        count += 1
        print(sub_name[sub]+"/"+ROI_name[roi]+"done!!!!!") 