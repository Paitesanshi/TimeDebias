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

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


class IPSV(nn.Module):
    r"""
        MF model
    """

    def __init__(self, config,psmodel):
        super(IPSV, self).__init__()

        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss

        # self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        # self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = nn.MSELoss()
        self.gamma=config['gamma_v']
        self.psmodel=psmodel
        self.user_embedding = torch.load('init_ps/user_embedding.pth')
        self.item_embedding = torch.load('init_ps/item_embedding.pth')
        #self.embedding_size = self.user_embedding.shape[1]
        self.mlp=MLPLayers([self.embedding_size*2,self.embedding_size,1],activation='sigmoid')
        # self.invP.weight = torch.nn.Parameter(ips_hat)
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

    def forward(self, user,item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        input=torch.cat([user_e,item_e],dim=1)
        w=self.mlp(input).squeeze()
        po=self.psmodel.get_p(user,item)
        po[po < 0.25] = 0.25
        invp=torch.reciprocal(po)
        # lowBound = torch.ones_like(invp) + (invp - torch.ones_like(invp)) / (torch.ones_like(InvP) * args.Gama[0])
        # upBound = torch.ones_like(invp) + (invp - torch.ones_like(invp)) * (torch.ones_like(InvP) * args.Gama[0])
        low=1+(invp-1)/self.gamma
        up=1+(invp-1)*self.gamma
        w.data *= (up - low)
        w.data+=low
        return w

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.sigmoid(self.forward(user, item))
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.sigmoid(self.forward(user, item))
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
