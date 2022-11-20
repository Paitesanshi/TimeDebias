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
import math

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class EqualPS(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(EqualPS, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.TIME = 'timestamp'
        self.K = config['K']
        self.sig = config['sig']
        # self.Ra
        self.RATING = config['RATING_FIELD']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.density = []
        # self.bu = torch.zeros(size=self.K)
        # self.bsig = torch.zeros(size=self.K)

        for i in range(self.K):
            density = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.Linear(self.embedding_size, 1),
                # nn.Sigmoid()
            )
            self.density.append(density)
            # self.bu[i] = i / len(self.bu)
        self.loss = nn.MSELoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

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

    def smooth(self, t, b):
        s = 1 + torch.exp(self.T * (t - b))
        s = 1 / s
        return s

    def forward(self, user, item,bid):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)

        ui = torch.cat((user_e, item_e), dim=-1)
        puit = self.density[bid](ui)
        return puit

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        # for i in range(len(user)):
        totloss=0
        for i in range(self.K):
            l=i/self.K
            r=(i+1)/self.K
            #mask=torch.where(((time>=l)&(time<=r)),1,0)
            mask=(time>=l)&(time<r)
            uk=user[mask]
            ik=item[mask]
            tk=time[mask]
            puit = self.forward(uk, ik,i).squeeze()
            loss = self.loss(puit, tk)
            totloss+=loss
        return totloss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        # time_origin = interaction[self.TIME_ORIGIN]
        pui=torch.zeros(size=(len(user),))
        for i in range(len(user)):
            bid=int(time[i]*self.K)
            if bid>=self.K:
                bid-=1
            pui[i]=self.forward(user[i],item[i],bid)
        # puit = self.forward(user, item)
        return pui
    def get_p(self,user,item,time):
        # pui = self.predict(interaction)
        pui = torch.zeros(size=(len(user),))
        for i in range(len(user)):
            bid = int(time[i] * self.K)
            if bid >= self.K:
                bid -= 1
            pui[i] = self.forward(user[i], item[i], bid)
        # pui = self.forward(user, item)
        po = -(time - pui) * (time - pui) / (2 * self.sig * self.sig)
        pt = 1 / (math.sqrt(2 * math.pi) * self.sig) * torch.exp(po)
        return pt

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        time = interaction[self.TIME]
        # time_origin = interaction[self.TIME_ORIGIN]
        puit = self.forward(user, all_item_e, time.unsqueeze(1))

        return puit
