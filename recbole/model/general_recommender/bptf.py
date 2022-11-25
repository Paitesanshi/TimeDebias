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
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


class BPTF(GeneralRecommender):
    r"""
        MF model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(BPTF, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        self.TIME = config['TIME_FIELD']
        # self.WDAY=config['WDAY_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_times=config['K']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.time_embedding = nn.Embedding(self.n_times, self.embedding_size)
        self.loss = nn.MSELoss(reduce=False)
        self.sigmoid = nn.Sigmoid()

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

    def get_time_embedding(self, time):
        # time=time*self.n_times
        # time=time.floor().long()
        # time = torch.where(time >=5, 4, time)
        return self.time_embedding(time)

    def forward(self, user, item,day):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        time_e = self.get_time_embedding(day.long())
        return torch.mul(torch.mul(user_e, item_e),time_e).sum(dim=1)


    def calculate_loss(self, interaction, weight=None):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        day=interaction[self.TIME].long()
        #rating = interaction[self.RATING]
        label = interaction[self.LABEL]
        output = self.forward(user, item,day)
        losses = self.loss(output, label)
        if weight != None:
            losses *= weight
        loss = torch.sum(losses) / len(losses)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        day = interaction[self.TIME].long()
        score = self.forward(user, item,day)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
