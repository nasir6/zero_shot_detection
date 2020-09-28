import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as LA

class ClsModel2(nn.Module):
    def __init__(self, in_channels):
        super(ClsModel2, self).__init__()
        print('__init__ model')
        
        att = np.load('/raid/mun/codes/zero_shot_detection/zsd/VOC/fasttext_synonym.npy')
        voc = np.load('/raid/mun/codes/zero_shot_detection/zsd/VOC/voc_fasttext.npy')
        
        att/=LA.norm(att, ord=2)
        # att = att[0:17]
        # voc/=LA.norm(voc, ord=2)
        print(f'att shape {att.shape}')
        # print(f'dictionary shape {voc.shape}')
        # self.D = torch.from_numpy(voc).type(torch.float).cuda()
        self.W = torch.from_numpy(att).type(torch.float).cuda()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=300, bias=True)
        # self.M = torch.nn.Parameter(data=torch.cuda.FloatTensor(self.W.shape[1], self.D.shape[0]), requires_grad=True).cuda()
        self.W.requires_grad = False
        # self.D.requires_grad = False

    def forward(self, feats):
        f = self.fc1(feats)

        x = f.mm(self.W.transpose(1, 0))
        # x = f.mm(torch.tanh(self.W.mm(self.M).mm(self.D)).transpose(1,0))
        return x