from sklearn.cluster import KMeans


import pandas as pd
import numpy as np
import torch
import math
from sklearn import model_selection
import time as tm
import numpy as np  # 数组相关的库
import matplotlib.pyplot as plt  # 绘图库
from scipy import stats
def calculate_caremers_v(df, column_a, column_b):
    """
    calculate carmer v for the 2 input columns in dataframe
    :param df: Pandas dataframe object
    :param column_a: 1st column to study
    :param column_b: 2nd column to study
    :return: Pandas dataframe object with the duplicated recorders removed.
    """
    if column_a not in df.columns:
        print("the input columne %s doesn't exit in the dataframe." % column_a)
        return None
    elif column_b not in df.columns:
        print("the input columne %s doesn't exit in the dataframe." % column_b)
        return None
    else:
        cross_tb = pd.crosstab(index=df[column_a], columns=df[column_b])
        np_tb = cross_tb.to_numpy()
        min_row_column = min(np_tb.shape[0], np_tb.shape[1])
        colume_sum = np_tb.sum(axis=0)
        row_sum = np_tb.sum(axis=1)
        total_sum = np_tb.sum()
        np_mid = np.matmul(row_sum.reshape(len(row_sum), 1),
                           colume_sum.reshape(1, len(colume_sum))) / total_sum
        new_tb = np.divide(np.power((np_tb - np_mid), np.array([2])),
                           np_mid)
        res = stats.chisquare(f_obs=np_tb,  # Array of obversed counts
                              f_exp=np_mid, axis=None, ddof=(np_tb.shape[0]- 1) * (np_tb.shape[1] - 1))
        print(res)
        return new_tb.sum() ,new_tb.sum() / (total_sum * (min_row_column - 1))
def clustering(fulldata_path):
    estimators = []
    f=open('netflix.log','w')
    for i in range(1, 20):
        estimators.append(('k_means_iris_' + str(i), KMeans(n_clusters=i)))
    data = pd.read_csv(fulldata_path, header=0,
                       names=[ 'item_id:token','user_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    totu=len(data['user_id:token'].value_counts())
    X=np.zeros(shape=(totu,1500))
    cnt={}
    uc={}
    ic={}
    nowi=0
    nowu=0
    for index, row in data.iterrows():
        u=row['user_id:token']
        i=row['item_id:token']
        if u not in cnt:
            cnt[u]=0
            uc[u]=nowu
            nowu+=1
        if i not in ic:
            ic[i]=nowi
            nowi+=1
        day = tm.localtime(int(row['timestamp:float'])).tm_wday
        row['timestamp:float']=day
        X[uc[u]][cnt[u]]=ic[i]
        cnt[u]+=1
    data.insert(data.shape[1], 'cluster', 0)
    for name, est in estimators:
        print("-----esimator name:",file=f)
        print(name,file=f)
        est.fit(X)
        kmeans_clustering_labels =est.labels_
        for index, row in data.iterrows():

            row['cluster']=kmeans_clustering_labels[uc[row['user_id:token']]]

        groups=list(data.groupby("cluster"))
        for c,d in groups:
            chi,v=calculate_caremers_v(d,'timestamp:float', 'rating:float')
            print("-----result:",file=f)
            print(c,chi,v,file=f)


if __name__ == '__main__':

    #clustering('ml-100k_origin.inter')
    #clustering('dataset/amazon_video_games/amazon_video_games.inter')
    clustering('../dataset/netflix/netflix.inter')