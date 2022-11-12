# -*- coding: utf-8 -*-
# @Time   : 2022/3/24
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
MF
################################################
Reference:
    Yehuda Koren et al, "Matrix factorization techniques for recommender systems"
"""

from audioop import bias
from genericpath import exists
import torch
import torch.nn as nn
import numpy as np
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender




class SVDPLUS(GeneralRecommender):
    r"""
    MF model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SVDPLUS, self).__init__(config, dataset)

        self.LABEL = config["LABEL_FIELD"]
        self.RATING = config['RATING_FIELD']
        self.Lambda=config['Lambda']
        # load parameters info
        self.embedding_size = config["embedding_size"]
        # define layers and loss
        self.pu = nn.Embedding(self.n_users, self.embedding_size)
        self.qi = nn.Embedding(self.n_items, self.embedding_size)
        self.y = nn.Embedding(self.n_items, self.embedding_size)
        self.bu = nn.Embedding(self.n_users,1)
        self.bi = nn.Embedding(self.n_items,1)
        # self.qi = nn.Embedding(self.n_items)
        self.sigmoid = nn.Sigmoid()
        (
            self.history,
            self.history_value,
            self.history_len,
        ) = dataset.history_item_matrix()

        self.avg_u = torch.mean(self.history_value, dim=1)
        self.avg = torch.mean(self.history_value)
        # self.bi = torch.zeros(self.n_items)
        # self.bu = torch.zeros(self.n_users)
        #self.alpha_u = nn.Embedding(self.n_users)
        # self.y = torch.ones((self.n_items, self.embedding_size))

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def getY(self, user):
        Ru = self.history[user]
        pos=self.history_len[user]
        I_Ru = self.history_len[user]
        sqrt_Ru = np.sqrt(I_Ru)
        #y_u = torch.zeros((len(user), self.embedding_size))

        user_impl_prf = torch.zeros((len(user), self.embedding_size))
        for i in range(len(user)):
            #y_u = torch.zeros(self.n_items)
            history_u=Ru[i][:pos[i]-1]
            if I_Ru[i].item() != 0:
                # for j in Ru[i]:
                #     if j==0:
                #         break
                #     #y_u[j]=self.y[j]
                user_impl_prf[i]=torch.sum(self.y(history_u),dim=0)/sqrt_Ru[i]
        return user_impl_prf,sqrt_Ru[i]

    def get_user_embedding(self, user):

        return self.pu(user)

    def get_item_embedding(self, item):

        return self.qi(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        b_u=self.bu(user).squeeze(dim=-1)
        b_i=self.bi(item).squeeze(dim=-1)
        u_y,_=self.getY(user)
        temp=torch.mul(item_e,user_e+u_y).sum(dim=1)
        pred = self.avg + b_u + b_i + temp
        # rating = self.avg_u[user] + self.bi[item] + self.bu[user] + torch.sum(
        #     item_e * (user_e + user_impl_prf))  # Формула оценки прогноза
        return pred

    def l2_loss(self,t):
        return torch.sum(t.pow(2))
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        rating = interaction[self.RATING]
        

        pred = self.forward(user, item)
        u_y,sqrt_u=self.getY(user)
        u_y=u_y*sqrt_u
        loss = (
            self.l2_loss(pred-rating)
            + 
            self.Lambda
            * (
                self.l2_loss(self.bu(user))
                + self.l2_loss(self.bi(item))
                + self.l2_loss(self.pu(user))
                + self.l2_loss(self.qi(item))
                + self.l2_loss(u_y)
            )
        )
        
        # loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        score = self.forward(user, item)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.qi.weight
        b_u = self.bu(user).squeeze(dim=-1)
        all_bi=self.bi.weight
        u_y, _ = self.getY(user)
        temp = torch.matmul( user_e + u_y,all_item_e.transpose(0, 1)).sum(dim=1)
        score = self.avg + b_u + all_bi + temp
        return score
