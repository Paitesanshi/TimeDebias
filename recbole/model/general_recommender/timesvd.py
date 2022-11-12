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




class TimeSVD(GeneralRecommender):
    r"""
    MF model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(TimeSVD, self).__init__(config, dataset)

        self.LABEL = config["LABEL_FIELD"]
        self.TIME = config['TIME_FIELD']
        self.lamb2 = 25
        self.lamb3 = 10
        self.lamb4 = 0.02 #0.02
        self.lamb5 = 0.01 #10
        self.lamb6 = 0.5 #15
        self.lamb7 = 0.1
        self.lamb8 = 0.1
        # load parameters info

        self.embedding_size = config["embedding_size"]
        self.item_bin_size = config["item_bin_size"]
        self.K=config['K']
        self.lr = config["lr"]
        self.Lambda = config["Lambda"]
        self.maxday_cat_code = 4096
        # define layers and loss
        self.pu = nn.Embedding(self.n_users, self.embedding_size)
        self.qi = nn.Embedding(self.n_items, self.embedding_size)
       
        self.sigmoid = nn.Sigmoid()
        (
            self.history,
            self.history_value,
            self.history_len,
        ) = dataset.history_item_matrix()
        self.avg_u = torch.mean(self.history_value, dim=1)
        self.avg = torch.mean(self.history_value)
        self.bi = torch.zeros(self.n_items)
        self.bu = torch.zeros(self.n_users)
        self.alpha_u = torch.zeros(self.n_users)
        self.y = torch.ones((self.n_items, self.embedding_size))
        self.tu = torch.zeros(self.n_users)

        self.bias_item_bin = torch.zeros((self.n_items, self.item_bin_size))
        self.time_matrix = (
            dataset.inter_matrix(form="csr", value_field="timestamp")
            .astype(np.float32)
            .toarray()
        )
        self.btday = torch.zeros(self.maxday_cat_code + 1)
        self.bcu = torch.zeros(self.n_users)
        self.wcu = torch.zeros(self.maxday_cat_code + 1)
        self.maxday_cat = torch.zeros(self.maxday_cat_code).long()
        self.max_time = self.time_matrix.max()
        for i in range(self.n_users):
            exist = self.time_matrix[i] != 0
            self.tu[i] = self.time_matrix[i].sum() / exist.sum()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def dev_u(self, user_id, time):

        tu = self.tu[user_id]
        signs=torch.where(time > tu, 1, -1)
        dev = signs * pow(abs(time - tu), 0.4)
        return dev

    def get_bin(self, time):
        # bin_size = self.max_time  / self.item_bin_size
        return time*self.K

    def getY(self, user, item):
        Ru = self.history[user]
        I_Ru = self.history_len[user]
        sqrt_Ru = np.sqrt(I_Ru)
        y_u = torch.zeros((len(user), self.embedding_size))

        user_impl_prf = torch.zeros((len(user), self.embedding_size))
        for i in range(len(user)):
            if I_Ru[i].item() == 0:
                user_impl_prf[i] = y_u[i]
            else:
                for j in Ru[i]:
                    y_u[i] += self.y[j]
                user_impl_prf[i] = y_u[i] / sqrt_Ru[i]
        return user_impl_prf, sqrt_Ru

    def get_user_embedding(self, user):

        return self.pu(user)

    def get_item_embedding(self, item):

        return self.qi(item)

    def forward(self, user, item, time):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        # user_impl_prf, sqrt_Ru = self.getY(user, item)
        dev = self.dev_u(user, time)
        alpha_value = self.alpha_u[user]
        bias_u = self.bu[user]
        bias_i = self.bi[item]
        self.bias_item_binvalue = self.get_bin(time)
        self.bias_user_tvalue = torch.mul(alpha_value, dev)
        bias_user_time = bias_u + self.bias_user_tvalue
        bias_user_time = bias_user_time + self.btday[user]
        bias_item_time = bias_i + self.bias_item_binvalue
        cui = self.bcu[user]
        cui=cui+self.wcu[self.maxday_cat[user]]
        bias_item_time = torch.mul(bias_item_time, cui)
        bias_vector = torch.mul(user_e, item_e).sum(dim=1)
        pred = self.avg + bias_user_time + bias_item_time + bias_vector
        # rating = self.avg_u[user] + self.bi[item] + self.bu[user] + torch.sum(
        #     item_e * (user_e + user_impl_prf))  # Формула оценки прогноза
        return pred

    def l2_loss(self,t):
        return t.pow(2)
    def calculate_loss(self, interaction,weight=None):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        time = interaction[self.TIME]


        pred = self.forward(user, item, time)
        losses = (
            self.l2_loss(pred-label)
            + 0.5
            * self.lamb4
            * (
                self.l2_loss(self.bu[user])
                + self.l2_loss(self.bi[item])
                + self.l2_loss(self.pu(user).sum(dim=1))
                + self.l2_loss(self.qi(item).sum(dim=1))
            )
            + 0.5 * self.lamb5 * self.l2_loss(self.bias_item_binvalue)
            + 0.5 * self.lamb6 * self.l2_loss(self.bias_user_tvalue)
            # + 0.5 * self.lamb7 * self.l2_loss(self.btday)
            + 0.5
            * self.lamb8
            * (self.l2_loss(self.wcu[self.maxday_cat[user]]) + self.l2_loss(self.bcu[user]))
        )
        if weight!=None:
            losses*=weight
        # loss = self.loss(output, label)
        loss=torch.sum(losses)
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
