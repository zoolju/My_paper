import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from semantic_fMRI import semantic_fMRI
from gnn import GNN
from element_wise_layer import Element_Wise_Layer
class fSGL(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim = 300):
        super(fSGL, self).__init__()
        self.seq = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True,dropout=0.3)
        self.num_classes = num_classes
        #self.transform_1 = nn.Linear(4854,2048)
        #self.transform_2 = nn.Linear(14,9)      
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.keep_prob = 0.7
        self.drop_layer = nn.Dropout(p=self.keep_prob)
        
# =============================================================================
#         self.word_semantic_image = semantic_image(num_classes= self.num_classes,
#                                       image_feature_dim = self.image_feature_dim,
#                                       word_feature_dim=self.word_feature_dim)
# =============================================================================
        self.word_semantic_fMRI = semantic_fMRI(num_classes = self.num_classes,
                                      voxel_num = self.image_feature_dim,
                                      word_feature_dim = self.word_feature_dim,
                                      keep_prob = self.keep_prob)

        self.word_features = word_features
        self._word_features = self.load_features()
        self.adjacency_matrix = adjacency_matrix
        self._in_matrix, self._out_matrix = self.load_matrix()
        self.time_step = time_step
        
        self.graph_net = GNN(input_dim=self.image_feature_dim,
                              time_step=self.time_step,
                              in_matrix=self._in_matrix,
                              out_matrix=self._out_matrix,
                              keep_prob = self.keep_prob)

        self.output_dim = output_dim
        self.fc_output = nn.Linear(2*self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, x,y,alpha,beta):
        batch_size = x.size()[0]
        fMRI_feature_map  = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map  = fMRI_feature_map[:,:,:]
        fMRI_feature_map,hidden = self.seq(fMRI_feature_map)
        fMRI_feature_map = torch.transpose(fMRI_feature_map,1,2)
        semantic_net_input = fMRI_feature_map
        semantic_net_input = torch.transpose(semantic_net_input,1,2)
        graph_net_input,coefficient = self.word_semantic_fMRI(batch_size,
                                     semantic_net_input,
                                     torch.tensor(self._word_features).cuda())
        graph_net_feature,_in_matrix = self.graph_net(graph_net_input)

        output = torch.cat((graph_net_feature.view(batch_size*self.num_classes,-1), graph_net_input.view(-1, self.image_feature_dim)), 1)
        output = self.fc_output(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result,voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)
        return result,voxel_weight,coefficient

    def load_features(self):
        return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()

    def load_matrix(self):
        mat = np.load(self.adjacency_matrix)
        #self.weight = Parameter(torch.Tensor(in_features, out_features))
        #mat = np.identity(57)
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
        _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
        return _in_matrix, _out_matrix

class DNN(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=57, word_feature_dim = 300):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(2048,2048)
        self.l2 = nn.Linear(2048,2048)

        self.num_classes = num_classes      
        self.fc_output = nn.Linear(2048, 2048)
        self.classifiers = Element_Wise_Layer(self.num_classes, 2048)

    def forward(self, x,y,alpha,beta):
        batch_size = y.size()[0]
        fMRI_feature_map = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map = fMRI_feature_map[:,5,:]
        fMRI_feature_map = F.relu(self.l1(fMRI_feature_map))
        fMRI_feature_map = F.relu(self.l2(fMRI_feature_map))
        output = self.fc_output(fMRI_feature_map)
        output = output.contiguous().view(batch_size, 1, 2048).repeat(1,self.num_classes,1)
        result,voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)
        return result,voxel_weight


class GRU(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=52, word_feature_dim = 300):
        super(GRU, self).__init__()
        self.GRU = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True,dropout=0.3)
        self.num_classes       = num_classes      
        self.image_feature_dim = image_feature_dim
        self.keep_prob         = 0.7
        self.drop_layer        = nn.Dropout(p=self.keep_prob)
        self.output_dim = output_dim
        self.fc_output = nn.Linear(self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, x,y,alpha,beta):
        batch_size = x.size()[0]
        fMRI_feature_map  = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map,hidden = self.GRU(fMRI_feature_map)
        fMRI_feature_map = torch.transpose(fMRI_feature_map,1,2)
        output = torch.sum(fMRI_feature_map,2).view(-1, 1,self.image_feature_dim).repeat(1,self.num_classes,1)
        output = self.fc_output(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result,voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)
        #print(torch.sigmoid(result))
        return result,voxel_weight,voxel_weight


class LSTM(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=57, word_feature_dim = 300):
        super(LSTM, self).__init__()
        self.seq = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True,dropout=0.5)
        self.num_classes = num_classes      
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.keep_prob = 0.5
        self.drop_layer = nn.Dropout(p=self.keep_prob)
        self.output_dim = output_dim
        self.fc_output = nn.Linear(self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, x,y,alpha,beta):
        batch_size = x.size()[0]
        fMRI_feature_map  = torch.tensor(y, dtype=torch.float32).cuda()
        fMRI_feature_map,hidden = self.seq(fMRI_feature_map)
        fMRI_feature_map = torch.transpose(fMRI_feature_map,1,2)
        output = torch.sum(fMRI_feature_map,2).view(-1, 1,self.image_feature_dim).repeat(1,self.num_classes,1)
        output = self.fc_output(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result,voxel_weight = self.classifiers(output)
        result = torch.sigmoid(result)
        return result,voxel_weight

