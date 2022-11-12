# # -*- coding: utf-8 -*-
# # @Time   : 2020/6/25
# # @Author : Shanlei Mu
# # @Email  : slmu@ruc.edu.cn
#
# # UPDATE:
# # @Time   : 2020/9/16
# # @Author : Shanlei Mu
# # @Email  : slmu@ruc.edu.cn
#
# r"""
# BPR
# ################################################
# Reference:
#     Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
# """
#
# import torch
# import torch.nn as nn
# import numpy as np
# from recbole.model.abstract_recommender import GeneralRecommender
# from recbole.model.init import xavier_normal_initialization
# from recbole.utils import InputType
# from recbole.model.layers import MLPLayers
# import time as tm
#
# def equivariant_layer(x, h_dim):
#     xm = torch.sum(x, dim=1, keepdims=True)
#     gamma=nn.Linear(x.shape[-1],h_dim)
#     lambd= nn.Linear(xm.shape[-1], h_dim,bias=False)
#     x=x.to(torch.float32)
#     xm = xm.to(torch.float32)
#     l_gamma = gamma(x)
#     l_lambda = lambd(xm)
#     out = l_gamma - l_lambda
#     return out
#
#
# class Inference(GeneralRecommender):
#     r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
#
#     """
#     input_type = InputType.POINTWISE
#
#     def __init__(self, config, dataset):
#         super(Inference, self).__init__(config, dataset)
#
#         # load parameters info
#         self.embedding_size = config['embedding_size']
#         self.sample_type = config['sample_type']
#         self.num_time_sample=config['num_time_sample']
#         self.batch_size=config['train_batch_size']
#         self.LABEL = config['LABEL_FIELD']
#         self.RATING = config['RATING_FIELD']
#         self.TIME='timestamp'
#         # define layers and loss
#         self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
#
#         self.g_logits=None
#         self.sigmoid = nn.Sigmoid()
#         self.vt_sample=torch.zeros(size=(self.batch_size,self.n_items,self.num_time_sample))
#         self.G_shared=torch.nn.Linear(5,self.embedding_size)
#         self.elu=nn.ELU()
#         self.nets=MLPLayers([self.embedding_size+self.num_time_sample,self.embedding_size,self.embedding_size,1])
#         self.ut_ed=torch.zeros(size=(self.n_users,))
#         self.ut_st=torch.zeros(size=(self.n_users,))
#         # parameters initialization
#         time_matrix = dataset.inter_matrix(form='coo',value_field=self.TIME).astype(np.float64).toarray()
#
#         for i in range(len(time_matrix)):
#             st = time_matrix[i].min()
#             ed = time_matrix[i].max()
#             self.ut_st[i]=st
#             self.ut_ed[i]=ed
#         # parameters initialization
#         self.apply(xavier_normal_initialization)
#
#     def get_vt_sample(self, user, item, time, time_origin=None):
#         batch_size = len(user)
#         vt_sample = torch.rand(batch_size, self.n_items, self.num_time_sample)
#
#         if self.sample_type == 1:
#
#             step = (self.ut_ed[user] - self.ut_st[user]) / self.num_time_sample
#             vt_sample *= step
#             for i in range(batch_size):
#                 for k in range(self.num_time_sample):
#                     #         self.vt_sample[i, :, k] = self.ut_st[user[i].item()] + k * step[user[i].item()]
#                     vt_sample[i, :, k] += self.ut_st[user[i].item()] + k * step[user[i].item()]
#             # vt_sample =self.vt_sample+ self.vt_fix[batch_size]
#             factual_t_position = ((time - self.ut_st[user]) / step).long()
#             factual_t_position = torch.where(factual_t_position >= self.num_time_sample, self.num_time_sample - 1,
#                                              factual_t_position)
#             # item=item.long()
#             vt_sample[range(len(user)), item, factual_t_position] = time
#
#         elif self.sample_type == 2:
#
#             for i in range(len(user)):
#                 time_origin[i] = tm.localtime(int(time_origin[i])).tm_hour
#             time_origin = time_origin.long()
#             step = self.hour_step
#             vt_sample *= step
#             vt_sample = vt_sample + self.vt_fix[:batch_size]
#             factual_t_position = time_origin
#
#             vt_sample[range(len(user)), item, factual_t_position] = time
#         elif self.sample_type == 3:
#
#             for i in range(len(user)):
#                 time_origin[i] = tm.localtime(int(time_origin[i])).tm_wday
#             time_origin = time_origin.long()
#             step = self.day_step
#             vt_sample *= step
#             vt_sample = vt_sample + self.vt_fix[:batch_size]
#             factual_t_position = time_origin
#
#             vt_sample[range(len(user)), item, factual_t_position] = time.cpu()
#         elif self.sample_type == 4:
#             pass
#             # entro=0
#
#         else:
#             factual_t_position = np.random.randint(self.num_time_sample, size=[len(user)])
#             vt_sample[range(len(user)), item, factual_t_position] = time
#         vt_mask = torch.zeros(size=[len(user), self.n_items,
#                                     self.num_time_sample]).to(self.device)
#         vt_mask[range(len(user)), item, factual_t_position] = 1
#         return vt_sample, vt_mask
#     # def get_vt_sample(self,user,item,time):
#     #     batch_size=len(user)
#     #     vt_sample = torch.rand(batch_size, self.n_items, self.num_time_sample)
#     #
#     #
#     #
#     #     if self.sample_type==1:
#     #
#     #         step=(self.ut_ed[user]-self.ut_st[user])/self.num_time_sample
#     #         # for i in range(self.n_users):
#     #         #     for j in range(self.n_items):
#     #         #         for k in range(self.num_time_sample):
#     #         #             self.vt_sample[i, j, k] = self.vt_sample[i, j, k] *set[i]
#     #         for i in range(len(user)):
#     #             for j in range(self.n_items):
#     #                 for k in range(self.num_time_sample):
#     #                     self.vt_sample[i, j, k] = self.vt_sample[i, j, k] *step[user[i].item()]+self.ut_st[user[i].item()]+(k-1)*step[user[i].item()]
#     #         factual_t_position =((time-self.ut_st[user])/step).long()
#     #         factual_t_position = torch.where(factual_t_position >= self.num_time_sample,self.num_time_sample-1 , factual_t_position)
#     #         #item=item.long()
#     #         self.vt_sample[range(len(user)), item, factual_t_position] = time
#     #
#     #     elif self.sample_type==2:
#     #         pass
#     #     else:
#     #         vt_sample = torch.rand(batch_size,self.n_items, self.num_time_sample)
#     #         factual_t_position = np.random.randint(self.num_time_sample, size=[len(user)])
#     #         self.vt_sample[range(len(user)), item, factual_t_position] = time
#     #     vt_mask = np.zeros(shape=[len(user), self.n_items,
#     #                               self.num_time_sample])
#     #     vt_mask[range(len(user)), item, factual_t_position] = 1
#     #     vt_mask = torch.from_numpy(vt_mask)
#     #     return  vt_sample,vt_mask
#
#     def get_data_sample(self,user,item,time):
#         user_single=torch.unique(user,sorted=True)
#         batch_size=len(user_single)
#         vt_sample = torch.rand(self.n_users, self.n_items, self.num_time_sample)
#
#         if self.sample_type==1:
#
#             step=(self.ut_ed[user]-self.ut_st[user])/self.num_time_sample
#             # for i in range(self.n_users):
#             #     for j in range(self.n_items):
#             #         for k in range(self.num_time_sample):
#             #             self.vt_sample[i, j, k] = self.vt_sample[i, j, k] *set[i]
#             for i in range(len(user)):
#                 for j in range(self.n_items):
#                     for k in range(self.num_time_sample):
#                         vt_sample[i, j, k] = vt_sample[i, j, k] *step[user[i].item()]+self.ut_st[user[i].item()]+(k-1)*step[user[i].item()]
#             factual_t_position =((time-self.ut_st[user])/step).long()
#             factual_t_position = torch.where(factual_t_position >= self.num_time_sample,self.num_time_sample-1 , factual_t_position)
#             for i in range(len(user)):
#                 vt_sample[user[i].item(), item[i].item(), factual_t_position[i].item()] = time[i].item()
#             #
#             #item=item.long()
#             #self.vt_sample[range(len(user)), item, factual_t_position] = time
#
#         elif self.sample_type==2:
#             pass
#         else:
#
#             factual_t_position = np.random.randint(self.num_time_sample, size=[len(user)])
#             for i in range(len(user)):
#                 vt_sample[user[i].item(), item[i].item(), factual_t_position[i].item()] = time[i].item()
#             #self.vt_sample[range(len(user)), item, factual_t_position] = time
#         # vt_mask = np.zeros(shape=[batch_size, self.n_items,
#         #                           self.num_time_sample])
#         # vt_mask[range(batch_size), item, factual_t_position] = 1
#         # vt_mask = torch.from_numpy(vt_mask)
#         return  vt_sample,user_single,factual_t_position
#
#
#
#
#     def get_user_embedding(self, user):
#         r""" Get a batch of user embedding tensor according to input user's id.
#
#         Args:
#             user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]
#
#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
#         """
#         return self.user_embedding(user)
#
#     def get_item_embedding(self, item):
#         r""" Get a batch of item embedding tensor according to input item's id.
#
#         Args:
#             item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]
#
#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
#         """
#         return self.item_embedding(item)
#
#     def forward(self, user, vt_sample):
#         user_e = self.get_user_embedding(user)
#         i_vt_outcome=dict()
#         for item in range(self.n_items):
#             t_cf=dict()
#             vt=vt_sample[:,item]
#             for index in range(self.num_time_sample):
#                 #t_sample=vt.unsqueeze(-1)
#                 input_cf_time=torch.cat([user_e,vt],1)
#                 output=self.nets(input_cf_time)
#                 t_cf[index]=output
#             i_vt_outcome[item] = torch.cat(list(t_cf.values()), dim=-1)
#         i_logits = torch.cat(list(i_vt_outcome.values()), dim=-1)
#         i_logits = torch.reshape(i_logits, shape=(-1, self.n_items, self.num_time_sample))
#         return i_logits,i_vt_outcome
#
#     def calculate_loss(self, interaction):
#         user = interaction[self.USER_ID]
#         item = interaction[self.ITEM_ID]
#         rating=interaction[self.RATING]
#         time=interaction[self.TIME]
#         self.vt_sample,self.vt_mask = self.get_vt_sample(user,item,time)
#         self.i_logits,self.i_outcome=self.forward(user,self.vt_sample)
#         i_logit_factual = torch.unsqueeze(torch.sum(self.vt_mask * self.i_logits, axis=[1, 2]), axis=-1)
#         i_loss_r = torch.sqrt(torch.mean((rating - i_logit_factual) ** 2))+torch.sqrt(torch.mean((self.g_logits - self.i_logits) ** 2))
#         return i_loss_r
#
#     def get_logits(self,interaction):
#         user = interaction[self.USER_ID]
#         item = interaction[self.ITEM_ID]
#         rating=interaction[self.RATING]
#         time=interaction[self.TIME]
#         vt_sample,user_single,factual_t_position = self.get_data_sample(user,item,time)
#         i_logits,i_outcome=self.forward(user_single,vt_sample)
#         i_logits=i_logits*5
#         for i in range(len(user)):
#                 i_logits[user[i].item(), item[i].item(), factual_t_position[i].item()] = rating[i].item()
#         i_logits=torch.ceil(i_logits)
#         return i_logits,vt_sample
#
#     def predict(self, interaction):
#         user = interaction[self.USER_ID]
#         item = interaction[self.ITEM_ID]
#         rating=interaction[self.RATING]
#         time=interaction[self.TIME]
#         self.vt_sample,self.vt_mask = self.get_vt_sample(user,item,time)
#         self.i_logits,self.i_outcome=self.forward(user,self.vt_sample)
#         i_logit_factual = torch.unsqueeze(torch.sum(self.vt_mask * self.i_logits, axis=[1, 2]), axis=-1)
#         return i_logit_factual
#
#     def full_sort_predict(self, interaction):
#         user = interaction[self.USER_ID]
#         user_e = self.get_user_embedding(user)
#         all_item_e = self.item_embedding.weight
#         score = torch.matmul(user_e, all_item_e.transpose(0, 1))
#         return score.view(-1)
# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn
import numpy as np
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
import time as tm


def equivariant_layer(x, h_dim, device):
    xm = torch.sum(x, dim=1, keepdims=True)
    gamma = nn.Linear(x.shape[-1], h_dim).to(device)
    lambd = nn.Linear(xm.shape[-1], h_dim, bias=False).to(device)
    x = x.to(torch.float32)
    xm = xm.to(torch.float32)
    l_gamma = gamma(x)
    l_lambda = lambd(xm)
    out = l_gamma - l_lambda
    return out


class Inference(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Inference, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.sample_type = config['sample_type']
        self.num_time_sample = config['num_time_sample']
        self.batch_size = config['train_batch_size']
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        self.TIME = 'timestamp'
        self.TIME_ORIGIN = 'time_origin'
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)

        self.model_g = None
        self.sigmoid = nn.Sigmoid()
        self.vt_sample = torch.zeros(size=(self.batch_size, self.n_items, self.num_time_sample))
        self.vt_fix = torch.zeros(size=(self.batch_size, self.n_items, self.num_time_sample))
        self.G_shared = torch.nn.Linear(5, self.embedding_size)
        self.elu = nn.ELU()
        self.nets = MLPLayers([self.embedding_size + self.num_time_sample, self.embedding_size, self.embedding_size, 1])
        self.ut_ed = torch.zeros(size=(self.n_users,))
        self.ut_st = torch.zeros(size=(self.n_users,))
        # parameters initialization
        time_matrix = dataset.inter_matrix(form='coo', value_field=self.TIME).astype(np.float64).toarray()
        self.day_timestamp = 86400
        self.hour_timestamp = 3600
        for i in range(len(dataset.inter_feat['timestamp'])):
            if dataset.inter_feat['timestamp'][i+1]!=dataset.inter_feat['timestamp'][i]:
                d1=i+1
                d2=i
                break
        # self.day_step = ((dataset.inter_feat['timestamp'][1] - dataset.inter_feat['timestamp'][0]) * self.day_timestamp) / (
        #             dataset.inter_feat['time_origin'][1] - dataset.inter_feat['time_origin'][0])
        # self.hour_step = ((dataset.inter_feat['timestamp'][1] - dataset.inter_feat['timestamp'][0]) * self.hour_timestamp) / (
        #         dataset.inter_feat['time_origin'][1] - dataset.inter_feat['time_origin'][0])
        self.day_step = ((dataset.inter_feat['timestamp'][d1] - dataset.inter_feat['timestamp'][d2]) * self.day_timestamp) / (
                    dataset.inter_feat['time_origin'][d1] - dataset.inter_feat['time_origin'][d2])
        self.hour_step = ((dataset.inter_feat['timestamp'][d1] - dataset.inter_feat['timestamp'][d2]) * self.hour_timestamp) / (
                dataset.inter_feat['time_origin'][d1] - dataset.inter_feat['time_origin'][d2])
        # self.day_step = ((dataset.inter_feat['timestamp'][1] - dataset.inter_feat['timestamp'][
        #     0]) * self.day_timestamp) / (
        #                         dataset.inter_feat['time_origin'][1] - dataset.inter_feat['time_origin'][0])
        # self.hour_step = ((dataset.inter_feat['timestamp'][1] - dataset.inter_feat['timestamp'][
        #     0]) * self.hour_timestamp) / (
        #                          dataset.inter_feat['time_origin'][1] - dataset.inter_feat['time_origin'][0])
        for i in range(len(time_matrix)):
            st = time_matrix[i].min()
            ed = time_matrix[i].max()
            self.ut_st[i] = st
            self.ut_ed[i] = ed
        if self.sample_type == 1:
            # step = (self.ut_ed[user] - self.ut_st[user]) / self.num_time_sample
            # for i in range(self.batch_size):
            #     for k in range(self.num_time_sample):
            #         self.vt_sample[i, :, k] = self.ut_st[user[i].item()] + k * step[user[i].item()]
            pass
        elif self.sample_type == 2:
            step = self.hour_step
            for k in range(self.num_time_sample):
                self.vt_fix[:, :, k] = k * step
        elif self.sample_type == 3:
            step = self.day_step
            for k in range(self.num_time_sample):
                self.vt_fix[:, :, k] = k * step
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_vt_sample(self, user, item, time, time_origin=None):
        batch_size = len(user)
        vt_sample = torch.rand(batch_size, self.n_items, self.num_time_sample)

        if self.sample_type == 1:

            step = (self.ut_ed[user] - self.ut_st[user]) / self.num_time_sample
            vt_sample *= step
            for i in range(batch_size):
                for k in range(self.num_time_sample):
                    #         self.vt_sample[i, :, k] = self.ut_st[user[i].item()] + k * step[user[i].item()]
                    vt_sample[i, :, k] += self.ut_st[user[i].item()] + k * step[user[i].item()]
            # vt_sample =self.vt_sample+ self.vt_fix[batch_size]
            factual_t_position = ((time - self.ut_st[user]) / step).long()
            factual_t_position = torch.where(factual_t_position >= self.num_time_sample, self.num_time_sample - 1,
                                             factual_t_position)
            # item=item.long()
            vt_sample[range(len(user)), item, factual_t_position] = time

        elif self.sample_type == 2:

            for i in range(len(user)):
                time_origin[i] = tm.localtime(int(time_origin[i])).tm_hour
            time_origin = time_origin.long()
            step = self.hour_step
            vt_sample *= step
            vt_sample = vt_sample + self.vt_fix[:batch_size]
            factual_t_position = time_origin

            vt_sample[range(len(user)), item, factual_t_position] = time
        elif self.sample_type == 3:

            for i in range(len(user)):
                time_origin[i] = tm.localtime(int(time_origin[i])).tm_wday
            time_origin = time_origin.long()
            step = self.day_step
            vt_sample *= step
            vt_sample = vt_sample + self.vt_fix[:batch_size]
            factual_t_position = time_origin

            vt_sample[range(len(user)), item, factual_t_position] = time.cpu()
        elif self.sample_type == 4:
            pass
            # entro=0

        else:
            factual_t_position = np.random.randint(self.num_time_sample, size=[len(user)])
            vt_sample[range(len(user)), item, factual_t_position] = time
        vt_mask = torch.zeros(size=[len(user), self.n_items,
                                    self.num_time_sample]).to(self.device)
        vt_mask[range(len(user)), item, factual_t_position] = 1
        return vt_sample, vt_mask

    # def get_vt_sample(self,user,item,time):
    #     batch_size=len(user)
    #     vt_sample = torch.rand(batch_size, self.n_items, self.num_time_sample)

    #     if self.sample_type==1:

    #         step=(self.ut_ed[user]-self.ut_st[user])/self.num_time_sample
    #         # for i in range(self.n_users):
    #         #     for j in range(self.n_items):
    #         #         for k in range(self.num_time_sample):
    #         #             self.vt_sample[i, j, k] = self.vt_sample[i, j, k] *set[i]
    #         for i in range(len(user)):
    #             for j in range(self.n_items):
    #                 for k in range(self.num_time_sample):
    #                     self.vt_sample[i, j, k] = self.vt_sample[i, j, k] *step[user[i].item()]+self.ut_st[user[i].item()]+(k-1)*step[user[i].item()]
    #         factual_t_position =((time-self.ut_st[user])/step).long()
    #         factual_t_position = torch.where(factual_t_position >= self.num_time_sample,self.num_time_sample-1 , factual_t_position)
    #         #item=item.long()
    #         self.vt_sample[range(len(user)), item, factual_t_position] = time

    #     elif self.sample_type==2:
    #         pass
    #     else:
    #         vt_sample = torch.rand(batch_size,self.n_items, self.num_time_sample)
    #         factual_t_position = np.random.randint(self.num_time_sample, size=[len(user)])
    #         self.vt_sample[range(len(user)), item, factual_t_position] = time
    #     vt_mask = np.zeros(shape=[len(user), self.n_items,
    #                               self.num_time_sample])
    #     vt_mask[range(len(user)), item, factual_t_position] = 1
    #     vt_mask = torch.from_numpy(vt_mask)
    #     return  vt_sample,vt_mask

    def get_data_sample(self, user, item, time, time_origin):
        user_single = torch.unique(user, sorted=True)
        batch_size = len(user_single)
        vt_sample = torch.rand(self.n_users, self.n_items, self.num_time_sample)
        if self.sample_type == 1:

            step = (self.ut_ed[user] - self.ut_st[user]) / self.num_time_sample
            vt_sample *= step
            for i in range(batch_size):
                for k in range(self.num_time_sample):
                    #         self.vt_sample[i, :, k] = self.ut_st[user[i].item()] + k * step[user[i].item()]
                    vt_sample[i, :, k] += self.ut_st[user[i].item()] + k * step[user[i].item()]
            # vt_sample =self.vt_sample+ self.vt_fix[batch_size]
            factual_t_position = ((time - self.ut_st[user]) / step).long()
            factual_t_position = torch.where(factual_t_position >= self.num_time_sample, self.num_time_sample - 1,
                                             factual_t_position)
            # item=item.long()
            vt_sample[range(len(user)), item, factual_t_position] = time

        elif self.sample_type == 2:

            for i in range(len(user)):
                time_origin[i] = tm.localtime(int(time_origin[i])).tm_hour
            time_origin = time_origin.long()
            step = self.hour_step
            vt_sample *= step
            vt_sample = vt_sample + self.vt_fix[:batch_size]
            factual_t_position = time_origin
            for i in range(len(user)):
                vt_sample[user[i].item(), item[i].item(), factual_t_position[i].item()] = time[i].item()
            #vt_sample[range(len(user)), item, factual_t_position] = time
        elif self.sample_type == 3:

            for i in range(len(user)):
                time_origin[i] = tm.localtime(int(time_origin[i])).tm_wday
            time_origin = time_origin.long()
            step = self.day_step
            vt_sample *= step
            vt_sample = vt_sample + self.vt_fix[:batch_size]
            factual_t_position = time_origin
            for i in range(len(user)):
                vt_sample[user[i].item(), item[i].item(), factual_t_position[i].item()] = time[i].item()
                #vt_sample[range(len(user)), item, factual_t_position] = time.cpu()
        elif self.sample_type == 4:
            pass
            # entro=0

        else:
            factual_t_position = np.random.randint(self.num_time_sample, size=[len(user)])
            vt_sample[range(len(user)), item, factual_t_position] = time
        # if self.sample_type==1:

        #     step=(self.ut_ed[user]-self.ut_st[user])/self.num_time_sample
        #     # for i in range(self.n_users):
        #     #     for j in range(self.n_items):
        #     #         for k in range(self.num_time_sample):
        #     #             self.vt_sample[i, j, k] = self.vt_sample[i, j, k] *set[i]
        #     for i in range(len(user)):
        #         for j in range(self.n_items):
        #             for k in range(self.num_time_sample):
        #                 vt_sample[i, j, k] = vt_sample[i, j, k] *step[user[i].item()]+self.ut_st[user[i].item()]+(k-1)*step[user[i].item()]
        #     factual_t_position =((time-self.ut_st[user])/step).long()
        #     factual_t_position = torch.where(factual_t_position >= self.num_time_sample,self.num_time_sample-1 , factual_t_position)
        #     for i in range(len(user)):
        #         vt_sample[user[i].item(), item[i].item(), factual_t_position[i].item()] = time[i].item()
        #     #
        #     #item=item.long()
        #     #self.vt_sample[range(len(user)), item, factual_t_position] = time

        # elif self.sample_type==2:
        #     pass
        # else:

        #     factual_t_position = np.random.randint(self.num_time_sample, size=[len(user)])
        #     for i in range(len(user)):
        #         vt_sample[user[i].item(), item[i].item(), factual_t_position[i].item()] = time[i].item()
        # self.vt_sample[range(len(user)), item, factual_t_position] = time
        # vt_mask = np.zeros(shape=[batch_size, self.n_items,
        #                           self.num_time_sample])
        # vt_mask[range(batch_size), item, factual_t_position] = 1
        # vt_mask = torch.from_numpy(vt_mask)
        return vt_sample, user_single, factual_t_position

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, vt_sample):
        user_e = self.get_user_embedding(user)
        i_vt_outcome = dict()
        for item in range(self.n_items):
            t_cf = dict()
            vt = vt_sample[:, item]
            vt = vt.to(self.device)
            for index in range(self.num_time_sample):
                # t_sample=vt.unsqueeze(-1)
                input_cf_time = torch.cat([user_e, vt], 1)
                output = self.nets(input_cf_time)
                t_cf[index] = output
            i_vt_outcome[item] = torch.cat(list(t_cf.values()), dim=-1)
        i_logits = torch.cat(list(i_vt_outcome.values()), dim=-1)
        i_logits = torch.reshape(i_logits, shape=(-1, self.n_items, self.num_time_sample))
        return i_logits, i_vt_outcome

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        rating = interaction[self.RATING]
        time = interaction[self.TIME]
        time_origin = interaction[self.TIME_ORIGIN]
        self.vt_sample, self.vt_mask = self.get_vt_sample(user, item, time, time_origin)
        self.vt_sample=self.vt_sample*100
        self.i_logits, self.i_outcome = self.forward(user, self.vt_sample)
        g_logits,_=self.model_g(user,item,rating,time,self.vt_sample)
        i_logit_factual = torch.unsqueeze(torch.sum(self.vt_mask * self.i_logits, axis=[1, 2]), axis=-1)
        i_loss_r = torch.sqrt(torch.mean((rating - i_logit_factual) ** 2)) + torch.sqrt(
            torch.mean((g_logits - self.i_logits) ** 2))
        return i_loss_r

    def get_logits(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        rating = interaction[self.RATING]
        time = interaction[self.TIME]
        time_origin = interaction[self.TIME_ORIGIN]
        vt_sample, user_single, factual_t_position = self.get_data_sample(user, item, time, time_origin)
        i_logits, i_outcome = self.forward(user_single, vt_sample)
        i_logits = i_logits * 5
        for i in range(len(user)):
            i_logits[user[i].item() , item[i].item(), factual_t_position[i].item()] = rating[i].item()
        i_logits = torch.ceil(i_logits)
        return i_logits, vt_sample

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        rating = interaction[self.RATING]
        time = interaction[self.TIME]
        time_origin = interaction[self.TIME_ORIGIN]
        self.vt_sample, self.vt_mask = self.get_vt_sample(user, item, time, time_origin)
        self.i_logits, self.i_outcome = self.forward(user, self.vt_sample)
        i_logit_factual = torch.unsqueeze(torch.sum(self.vt_mask * self.i_logits, axis=[1, 2]), axis=-1)
        return i_logit_factual

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
