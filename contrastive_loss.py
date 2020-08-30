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
        self.att = np.load('/raid/mun/codes/zero_shot_detection/zsd_abl/MSCOCO/fasttext.npy')
        # self.att/=LA.norm(self.att, ord=2)

        device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))

        self.att = torch.from_numpy(self.att).to(device)
        self.temperature = 0.05

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
        
        # dist = (dot_prod_feats - dot_prod_att) * (dot_prod_feats - dot_prod_att)
        # dist = torch.exp(dist / self.temperature)
        
        dist = torch.exp(torch.abs(dot_prod_feats - dot_prod_att) / self.temperature)
        pos = torch.sum(dist * mask_pos) / (mask_pos.sum() + 1e-6)
        neg = torch.sum(dist * mask_neg) / (mask_neg.sum() + 1e-6)

        # loss = torch.sum(abs_dist*mask_neg) / (mask_neg.sum() + 1e-6)
        # loss += torch.sum(abs_dist*mask_pos) / (mask_pos.sum() + 1e-6)

        loss = (- torch.log(pos / (pos + neg) )).mean()

        # loss = torch.sum(torch.abs(dot_prod_feats - dot_prod_att)*mask_neg) / (mask_neg.sum() + 1e-6)
        # loss = torch.sum(torch.square(dot_prod_feats - dot_prod_att)*mask_neg) / (mask_neg.sum() + 1e-6)
        # loss = 0.0 * (1.0 - pos_pairs_mean) + (1.0+ neg_pairs_mean)

        return loss




class ConLossReal(nn.Module):
    def __init__(self, features_mean):
        super(ConLossReal, self).__init__()
        self.device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))
        self.seen_feats_mean = F.normalize(torch.from_numpy(features_mean), p=2, dim=1).float().to(self.device)

    def forward(self, features, labels=None):
        

        # import pdb; pdb.set_trace()

        # normalize features   

        features = F.normalize(features, p=2, dim=1)

        dot_prod = torch.matmul(features, self.seen_feats_mean[labels].t())


        labels = labels[:, None] # extend dim

        mask = torch.eq(labels, labels.t()).byte().to(self.device)

        eye = torch.eye(mask.shape[0], mask.shape[1]).byte().to(self.device)
        
        mask_pos = mask.masked_fill(eye, 0).float()

        mask_neg = (~mask).float()
        # mask_neg.masked_fill_(eye, 0)
        

        # q=mask_pos * dot_prod
        # q[q>0.89]=1.0


        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (1.0+ neg_pairs_mean)

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

        loss = (1.0 - pos_pairs_mean) + (1.0+ neg_pairs_mean)

        return loss






























# class SupConLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature

#     def forward(self, features, labels=None, mask=None):
#         """Compute loss for model. If both `labels` and `mask` are None,
#         it degenerates to SimCLR unsupervised loss:
#         https://arxiv.org/pdf/2002.05709.pdf

#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
#         features = F.normalize(features, p=2, dim=2)

#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)

#         batch_size = features.shape[0]
#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.t()).float().to(device)
#         else:
#             mask = mask.float().to(device)

#         import pdb; pdb.set_trace()

#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         if self.contrast_mode == 'one':
#             anchor_feature = features[:, 0]
#             anchor_count = 1
#         elif self.contrast_mode == 'all':
#             anchor_feature = contrast_feature
#             anchor_count = contrast_count
#         else:
#             raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.t()),
#             self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.ones_like(mask).scatter(
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask

#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()

#         return loss







