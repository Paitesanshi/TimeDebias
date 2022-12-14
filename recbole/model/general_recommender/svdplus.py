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
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


class SVDPLUS(GeneralRecommender):
    r"""
    MF model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SVDPLUS, self).__init__(config, dataset)

        self.task = config['task']
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        # self.TIME = config['TIME_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']
        # self.n_times = config['K']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.b_u = nn.Embedding(self.n_users, 1)
        self.b_i = nn.Embedding(self.n_items, 1)
        self.b = nn.Parameter(torch.Tensor(1))
        if self.task == 'ps':
            self.loss = nn.BCELoss(reduce='mean')
            self.sigmoid = nn.Sigmoid()
        else:
            self.loss = nn.MSELoss(reduce=False)
            self.sigmoid = None

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):

        return self.user_embedding(user)

    def get_item_embedding(self, item):

        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        # time_e = self.get_time_embedding(time.long())
        output = torch.mul(user_e, item_e).sum(dim=1)
        # output+=time_e.squeeze()+self.b+self.b_u(user).squeeze()+self.b_i(item).squeeze()
        output += self.b + self.b_u(user).squeeze() + self.b_i(item).squeeze()
        if self.sigmoid == None:
            return output
        output = self.sigmoid(output)
        return output

    def calculate_loss(self, interaction, weight=None):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # time = interaction[self.TIME]
        if self.task == 'ps':
            label = interaction[self.LABEL]
        else:
            label = interaction[self.RATING]

        # output = self.forward(user, item, time)
        output = self.forward(user, item)
        loss = self.loss(output, label)
        if weight != None:
            loss *= weight
        loss = torch.sum(loss) / len(loss)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item)
        return score

    # def get_p(self,user,item,time):
    def get_p(self, user, item):
        output = self.forward(user, item)
        return output

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
