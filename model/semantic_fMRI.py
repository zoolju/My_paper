import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch
import torchsnooper


class semantic_fMRI(nn.Module):
    def __init__(self, num_classes, voxel_num, word_feature_dim,keep_prob, intermediary_dim = 2048):
        super(semantic_fMRI, self).__init__()
        self.num_classes = num_classes
        self.voxel_num = voxel_num
        self.keep_prob = keep_prob
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim
        self.drop_layer = nn.Dropout(p=self.keep_prob)
        self.fc_1_f = nn.Linear(self.voxel_num,        self.intermediary_dim, bias=False)
        self.fc_2_f = nn.Linear(self.word_feature_dim, self.intermediary_dim, bias=False)
        self.fc_3_f = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a_f = nn.Linear(self.intermediary_dim, self.voxel_num)
        #self.fc_1_f_1 = nn.Linear(1, self.num_classes)
    def forward(self,batch_size, img_feature_map, word_features):
        timepoints_num = img_feature_map.size()[1]
        #img_feature_map = torch.transpose(img_feature_map, 1, 2)
        #print(img_feature_map.shape)
        f_wh_feature = img_feature_map.contiguous().view(batch_size*timepoints_num, -1)
        #print(f_wh_feature.shape)
        f_wh_feature = self.fc_1_f(f_wh_feature).view(batch_size*timepoints_num, 1, -1).repeat(1, self.num_classes, 1)
        
# =============================================================================
#         f_wh_feature = self.fc_1_f(f_wh_feature).view(batch_size*timepoints_num, 1, -1)
#         print(f_wh_feature.shape)
#         f_wh_feature = self.fc_1_f_1(f_wh_feature)
# =============================================================================
        f_wh_feature_d = self.drop_layer(f_wh_feature)      
        f_wd_feature = self.fc_2_f(word_features).view(1, self.num_classes, self.intermediary_dim).repeat(batch_size*timepoints_num,1,1)
        f_wd_feature_d = self.drop_layer(f_wd_feature)        
        lb_feature  = self.fc_3_f(torch.tanh(f_wh_feature_d*f_wd_feature_d).view(-1,self.intermediary_dim))
        lb_feature  = self.drop_layer(lb_feature)
        #lb_feature  = torch.transpose(lb_feature.view(batch_size,timepoints_num,self.num_classes,self.intermediary_dim),1,3)
        coefficient = self.fc_a_f(lb_feature)
        coefficient = self.drop_layer(coefficient)
        coefficient = torch.transpose(coefficient.view(batch_size, timepoints_num, self.num_classes,-1),1,2)
        coefficient = F.softmax(coefficient, dim=2)
        coefficient = coefficient.view(batch_size, self.num_classes, timepoints_num,-1)
        coefficient = torch.transpose(coefficient,1,2)
        #img_feature_map = img_feature_map.view(batch_size,timepoints_num, 1, self.voxel_num).repeat(1,1, self.num_classes, 1)* coefficient
        img_feature_map = lb_feature.view(batch_size,timepoints_num, self.num_classes, self.voxel_num) * coefficient
        graph_net_input = torch.sum(img_feature_map,1)
        return graph_net_input,coefficient

