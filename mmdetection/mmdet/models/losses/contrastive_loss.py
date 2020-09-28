"""
Author: Nasir Hayat (nasirhayat6160@gmail.com)
Date: June 10, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from numpy import linalg as LA


class AttFeatsCon(nn.Module):
    def __init__(self):
        super(AttFeatsCon, self).__init__()
        self.att = np.load('../MSCOCO/fasttext.npy')
        self.att/=LA.norm(self.att, ord=2)

        device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))

        self.att = torch.from_numpy(self.att).to(device)

    def get_random_noise(self, bs, att_dim):
        """
        returns normal initialized noise tensor 
        """
        z = torch.cuda.FloatTensor(bs, att_dim)
        z.normal_(0, 1)
        return z

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # import pdb; pdb.set_trace()

        # normalize features   
        features = F.normalize(features, p=2, dim=1)

        att = self.att[labels]
        noise = self.get_random_noise(att.shape[0], att.shape[1])
        
        att = torch.cat((noise, att), 1)

        att = F.normalize(att, p=2, dim=1)


        labels = labels[:, None] # extend dim

        mask = torch.eq(labels, labels.t()).byte().to(device)

        eye = torch.eye(mask.shape[0], mask.shape[1]).byte().to(device)
        
        mask_pos = mask.masked_fill(eye, 0).float()

        mask_neg = (~mask).float()

        # mask_neg.masked_fill_(eye, 0)
        dot_prod_feats = torch.matmul(features, features.t())
        dot_prod_att = torch.matmul(att, att.t())
        

        # pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        # feats_neg_pairs = (mask_neg * dot_prod_feats).sum() / (mask_neg.sum() + 1e-6)
        # att_neg_pairs_mean = (mask_neg * dot_prod_att).sum() / (mask_neg.sum() + 1e-6)

        # loss = torch.abs((feats_neg_pairs_mean - att_neg_pairs_mean))

        # loss = torch.sum(torch.abs(dot_prod_feats - dot_prod_att)*mask_neg) / (mask_neg.sum() + 1e-6)
        diff = (dot_prod_feats - dot_prod_att)
        # diff = diff * diff
        loss = torch.sum((diff*diff)*mask_neg) / (mask_neg.sum() + 1e-6)
        # loss = 0.0 * (1.0 - pos_pairs_mean) + (1.0+ neg_pairs_mean)

        return loss


class SupConLoss(nn.Module):
    def __init__(self):
        super(SupConLoss, self).__init__()

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # import pdb; pdb.set_trace()

        # normalize features   
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None] # extend dim

        mask = torch.eq(labels, labels.t()).byte().to(device)

        eye = torch.eye(mask.shape[0], mask.shape[1]).byte().to(device)
        
        mask_pos = mask.masked_fill(eye, 0).float()

        mask_neg = (~mask).float()
        # mask_neg.masked_fill_(eye, 0)
        dot_prod = torch.matmul(features, features.t())
        

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = 0.0 * (1.0 - pos_pairs_mean) + (1.0+ neg_pairs_mean)

        return loss











