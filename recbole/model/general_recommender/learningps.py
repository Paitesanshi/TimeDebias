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


class LearningPS(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(LearningPS, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.TIME='timestamp'
        self.K=config['K']
        self.T = config['T']
        self.M=config['M']
        self.sig = config['sig']
        # self.Ra
        self.RATING = config['RATING_FIELD']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.density=nn.ModuleList()
        # self.bu = torch.nn.Parameter(torch.empty(size=(self.K,)),requires_grad=True)
        bb=torch.empty(size=(self.K-1,))
        self.bsig = torch.full(size=(self.K-1,),fill_value=self.sig,requires_grad=True).to(self.device)
        for i in range(self.K):
            sub_density = nn.Sequential(
                nn.Linear(self.embedding_size*2, self.embedding_size),
                nn.Linear(self.embedding_size, 1),
                nn.Sigmoid()
            )
            self.density.append(sub_density)
            # self.bu[i] = i / len(self.bu)
            if i<self.K-1:
                bb[i]=(i+1) / self.K
        self.bu = torch.nn.Parameter(bb, requires_grad=True).to(self.device)
        #self.bu[0].requires_grad_(False)
        # self.bu=bb
        #self.bu.requires_grad_(True)
        # for i in range(len(self.bu)):
        #     self.bu[i]=i/len(self.bu)
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

    def smooth(self,t,b):
        s=1+torch.exp(self.T*(t-b))
        s=1/s
        return s
    def forward(self, user, item,bid):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        ui=torch.cat((user_e,item_e),dim=-1)
        puit=self.density[bid](ui)
        return puit

    # def get_normal(self,mu,sig,x):
    def sample(self):
        r=torch.normal(0,1,size=(self.K-1,))
        pos=r*self.bsig+self.bu
        spos,_=torch.sort(pos)
        return spos
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        totloss = torch.tensor(0.)
        # for i in range(self.K):
        #     if i >= 1:
        #         # l = self.bu[i - 1]
        #         bl = torch.normal(self.bu[i - 1].data, self.bsig[i-1].data)
        #         #bl=torch.full((self.M,),self.bu[i-1])
        #     else:
        #         bl = torch.zeros(size=(self.M,))
        #     if i < self.K - 1:
        #         # r = self.bu[i]
        #         #br = torch.normal(self.bu[i].data, self.bsig[i].data, size=(self.M,))
        #         br = torch.full((self.M,), self.bu[i])
        #     else:
        #         br = torch.ones(size=(self.M,))
        #
        #     for j in range(self.M):
        #         mask = (time >= bl[j]) & (time <= br[j])
        #         uk = user[mask]
        #         if len(uk)==0:
        #             continue
        #         ik = item[mask]
        #         tk = time[mask]
        #         pui = self.forward(uk, ik, i).squeeze()
        #         s1=self.smooth(tk,bl[j])
        #         s2 = self.smooth(tk, br[j])
        #         s=s2-s1
        #         loss=0
        #         for k in range(len(uk)):
        #             loss+=(pui[k]-tk[k])*(pui[k]-tk[k])*s[k]
        #         loss/=len(uk)
        #         totloss += loss
        for j in range(self.M):
            pos=self.sample()
            for i in range(self.K):
                if i >= 1:
                    l = pos[i - 1]
                else:
                    l=0
                if i < self.K - 1:
                    r = pos[i]
                else:
                    r=1
                    #br = torch.ones(size=(self.M,))
                mask = (time >= l) & (time <= r)
                uk = user[mask]
                if len(uk)==0:
                    continue
                ik = item[mask]
                tk = time[mask]
                pui = self.forward(uk, ik, i).squeeze()
                s1=self.smooth(tk,l)
                s2 = self.smooth(tk, r)
                s=s2-s1
                # for k in range(len(uk)):
                #     loss+=(pui[k]-tk[k])*(pui[k]-tk[k])*s[k]
                loss=torch.sum((pui-tk)*(pui-tk)*s)
                loss/=len(uk)
                totloss += loss
        # for j in range(self.M):
        #     for i in range(self.K):
        #         if i >= 1:
        #             l = self.bu[i - 1]
        #         else:
        #             l=0
        #         if i < self.K - 1:
        #             r = self.bu[i]
        #         else:
        #             r=1
        #             #br = torch.ones(size=(self.M,))
        #         mask = (time >= l) & (time <= r)
        #         uk = user[mask]
        #         if len(uk)==0:
        #             continue
        #         ik = item[mask]
        #         tk = time[mask]
        #         pui = self.forward(uk, ik, i).squeeze()
        #         s1=self.smooth(tk,l)
        #         s2 = self.smooth(tk, r)
        #         s=s2-s1
        #         # for k in range(len(uk)):
        #         #     loss+=(pui[k]-tk[k])*(pui[k]-tk[k])*s[k]
        #         loss=torch.sum((pui-tk)*(pui-tk)*s)
        #         loss/=len(uk)
        #         totloss += loss
        totloss/=self.M

        return totloss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        # time_origin = interaction[self.TIME_ORIGIN]
        pui = torch.zeros(size=(len(user),))
        p=0
        for i in range(len(user)):
            while p<self.K-1 and time[i]>self.bu[p] :
                p+=1
            pui[i] = self.forward(user[i], item[i], p)
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
