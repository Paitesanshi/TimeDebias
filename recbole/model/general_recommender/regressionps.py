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



class RegressionPS(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RegressionPS, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.TIME='timestamp'
        self.K=config['K']
        self.T = config['T']
        self.sig=config['sig']
        # self.Ra
        self.invips=nn.Embedding(100,2)
        self.RATING = config['RATING_FIELD']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.density = nn.Sequential(
            nn.Linear(self.embedding_size*2, self.embedding_size),
            nn.Linear(self.embedding_size, 1),
            # nn.Sigmoid()
        )
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

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)

        ui=torch.cat((user_e,item_e),dim=-1)
        puit=self.density(ui).squeeze()
        return puit

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        #time_origin = interaction[self.TIME_ORIGIN]
        puit=self.forward(user,item)
        #loss=-(rating*torch.log(puit)+(1-rating)*torch.log(1-puit)).mean()
        loss=self.loss(puit,time)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        # time_origin = interaction[self.TIME_ORIGIN]
        puit = self.forward(user, item)
        return puit

    def get_p(self,user,item,time):
        # pui = self.predict(interaction)
        pui = self.forward(user, item)
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
