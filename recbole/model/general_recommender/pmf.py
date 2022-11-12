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


class PMF(GeneralRecommender):
    r"""
        MF model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(PMF, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
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

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        all_item_e = self.item_embedding.weight
        all_score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        all_score=torch.softmax(all_score,dim=1)
        #score = torch.zeros(len(user))
        # for i in range(len(user)):
        #     all_score = torch.mul(user_e[i], all_item_e).sum(1)
        #     all_score = torch.softmax(all_score,dim=0)
        #     # torch.mul(user_emb[index], item_emb_tot).sum(1)
        #     score[i]=all_score[item[i]]
        score=all_score.gather(1, item.unsqueeze(dim=1)).squeeze()
        return score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        rating=interaction[self.RATING]
        output = self.forward(user, item)
        loss = self.loss(output, rating)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item)
        # score=self.sigmoid(score)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
