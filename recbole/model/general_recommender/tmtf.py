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


class TMTF(GeneralRecommender):
    r"""
    MF model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(TMTF, self).__init__(config, dataset)
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        self.TIME = config['TIME_FIELD']
        self.WDAY = config['WDAY_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_times = config['K']
        self.b_T = nn.Embedding(self.n_periods, 1)
        self.b_u = nn.Embedding(self.n_users, 1)
        self.b_i = nn.Embedding(self.n_items, 1)
        self.b = nn.Parameter(torch.Tensor(1))
        self.sigmoid=nn.Sigmoid()
        self.apply(xavier_normal_initialization)



    def forward(self, user, item, time):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        time_e = self.time_embedding(time)
        if self.task.upper() == 'OIPT':
            # # u_v * (i_v + T_v) + b_T
            uit_e = torch.mul(user_e, time_e + item_e).sum(-1).float() + self.b_T(time).squeeze()
        else:
            # # u_v * (i_v + T_v) + b + b_i + b_u + b_T
            uit_e = torch.mul(user_e, time_e + item_e).sum(-1).float() + self.b + self.b_u(user).squeeze() + self.b_i(
                item).squeeze() + self.b_T(time).squeeze()
        # # u_v * i_v + u_v * T_v + i_v * T_v
        # uit_e = torch.mul(user_e, time_e + item_e).sum(-1).float() + torch.mul(item_e, time_e).sum(-1).float() + self.b_T(itemage).squeeze()
        # # v_i * (v_u + v_t) + b_T
        # uit_e = torch.mul(item_e, time_e + user_e).sum(-1).float() + self.b_T(itemage).squeeze()
        return self.sigmoid(uit_e)



    def calculate_loss(self, interaction, weight=None):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        time = interaction[self.WDAY]


        # loss = self.loss(output, label)
        loss = torch.sum(losses)/len(losses)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        score = self.forward(user, item, time)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
