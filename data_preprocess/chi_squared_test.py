import pandas as pd
import numpy as np

# np.set_printoptions(threshold=np.inf)
import torch
import math
from sklearn import model_selection
import time as tm
import numpy as np  # 数组相关的库
import matplotlib.pyplot as plt  # 绘图库
from scipy import stats
def split_unbiasedset_TO2(path='', per=0.7, sep='\t'):  # 将unbiased data划分为valid set和test set  3:7  (key)
    validset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float', 'intervene_mask:token'))
    testset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float', 'intervene_mask:token'))
    data = pd.read_csv(path, header=0, names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                                              'intervene_mask:token'], sep=sep)
    print('data', data)

    user = data['user_id:token'].unique()
    for u in user:
        u_data = data[data['user_id:token'].isin([u])]
        if (len(u_data) < 10):
            testset = testset.append(u_data)
            continue
        u_validdata, u_testdata = model_selection.train_test_split(u_data, test_size=per, shuffle=True)
        validset = validset.append(u_validdata)
        testset = testset.append(u_testdata)

    validset.to_csv("kuai_valid_no.csv", header=True, index=False, sep=sep)
    testset.to_csv("kuai_test_no.csv", header=True, index=False, sep=sep)
    print('dataset_num:', len(data))
    print('validset_num:', len(validset))
    print('testset_num:', len(testset))


def split_unbiasedset_TO3(unbias_path='', bias_path=''):  # 将unbiased data划分为train set,valid set和test set  1:1:2
    trainset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float','timestamp:float',
                 'intervene_mask:token'))
    validset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float','timestamp:float',
                 'intervene_mask:token'))
    testset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float',
                 'intervene_mask:token'))
    data = pd.read_csv(unbias_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float',
                              'intervene_mask:token'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    for u in user:
        u_data = data[data['user_id:token'].isin([u])]
        if (len(u_data) < 2):
            testset = testset.append(u_data)
            continue
        u_train_valid_data, u_test_data = model_selection.train_test_split(u_data, test_size=0.5, shuffle=True)
        if (len(u_train_valid_data) < 2):
            trainset = trainset.append(u_train_valid_data)
            continue
        u_train_data, u_valid_data = model_selection.train_test_split(u_train_valid_data, test_size=0.5, shuffle=True)
        trainset = trainset.append(u_train_data)
        validset = validset.append(u_valid_data)
        testset = testset.append(u_test_data)

    print('dataset_num:', len(data))
    print('trainset_num:', len(trainset))
    print('validset_num:', len(validset))
    print('testset_num:', len(testset))

    train_data = pd.read_csv(bias_path, header=0,
                             names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float',
                                    'intervene_mask:token'], sep='\t')
    print('train_data_num:', len(train_data))
    trainset = [train_data, trainset]
    trainset = pd.concat(trainset)
    print('trainset_num_concat:', trainset)
    print('trainset_num_concat:', len(trainset))

    trainset.to_csv("ml_train_mix.csv", header=True, index=False, sep='\t')
    validset.to_csv("ml_valid_mix.csv", header=True, index=False, sep='\t')
    testset.to_csv("ml_test_mix.csv", header=True, index=False, sep='\t')


def transform_df(path=''):  # 添加intervene_mask字段
    # data = pd.read_csv(path, header=0, usecols=[0, 1, 6, 7],
    #                   names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float'], sep=',')
    data = pd.read_csv(path, header=0, usecols=[0, 1, 2, 3],
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    print(data)
    data.insert(data.shape[1], 'intervene_mask:token', 'True')
    print(data)
    data.to_csv("ml_unbias_mask.csv", header=True, index=False, sep='\t')
    # data.to_csv("yahoo_unbias.csv", header=True, index=False, sep='\t')


def positive_filter(path=''):  # kuai中筛选正样本
    data = pd.read_csv(path, header=0, usecols=[0, 1, 2, 3, 4],
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                              'intervene_mask:token'],
                       sep='\t')
    data_filtered = data[(data['watch_ratio:float'] >= 2)]
    data_filtered.to_csv("kuai_unbias_filter.csv", header=True, index=False, sep='\t')
    print(data_filtered)


def sampling_weight(path=''):
    data = pd.read_csv(path, header=0, names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'],
                       sep='\t')
    print('data\n', data)

    count = data['item_id:token'].value_counts()
    print('count\n', count)

    id, num = np.unique(count.values, return_counts=True)
    print('频数', id)
    print('次数', num)

    max_count = count.max()
    relative_count = max_count / count
    print('relative_count\n', relative_count)

    popularity = count[data['item_id:token']]
    weight = relative_count[data['item_id:token']]
    print('popularity', popularity)
    print('weight', weight)

    popularity = popularity.values
    weight = weight.values
    data.insert(data.shape[1], 'popularity', popularity)
    data.insert(data.shape[1], 'weight', weight)
    print('data\n', data)

    data_samples = data.sample(frac=0.2, random_state=2022, weights='weight')
    print('data_samples\n', data_samples)
    print('data_samples[pop]', data_samples['popularity'])

    id, num = np.unique(data['popularity'], return_counts=True)
    print('频数', id)
    print('次数', num)

    id, num = np.unique(data_samples['popularity'], return_counts=True)
    # print('频数', id)
    print('次数', num)


def judge_userID(train_path='', test_path=''):  # 判断训练集中的用户是否都有测试数据
    train_data = pd.read_csv(train_path, header=0,
                             names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'],
                             sep='\t')
    test_data = pd.read_csv(test_path, header=0,
                            names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'],
                            sep='\t')
    #print('train_data',train_data)
    #print('test_data', test_data)
    id, num = np.unique(train_data['user_id:token'], return_counts=True)
    print('id1', id.shape)
    id, num = np.unique(test_data['user_id:token'], return_counts=True)
    print('id2', id.shape)


# 以下三个函数用于生成半合成数据集
def split_fulldata_TO2(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)
    trainset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'))
    testset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'))
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    for u in user:
        u_data = data[data['user_id:token'].isin([u])]
        # ml:len(u_data)最小为20
        u_train_data, u_test_data = model_selection.train_test_split(u_data, test_size=0.5, shuffle=True)
        trainset = trainset.append(u_train_data)
        testset = testset.append(u_test_data)
    print('dataset_num:', len(data))
    print('trainset_num:', len(trainset))
    print('testset_num:', len(testset))

    trainset.to_csv("ml_train_bias.csv", header=True, index=False, sep='\t')
    testset.to_csv("ml_test_bias.csv", header=True, index=False, sep='\t')


# def sampling_from_biasData(full_path='', bias_path=''):  # 从整体数据计算流行度，并基于此，从bias_data中采样50%作为unbias_data  (key)
#     full_data = pd.read_csv(full_path, header=0,
#                             names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'],
#                             sep='\t')
#     print('full_data\n', full_data)
#     bias_data = pd.read_csv(bias_path, header=0,
#                             names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'],
#                             sep='\t')
#     print('bias_data\n', bias_data)
#
#     count={}
#     max_count=0
#     x=[]
#     y=[]
#     full_data=full_data.sort_values(by='item_id:token')
#     for index, row in full_data.iterrows():
#         day=tm.localtime(int(row['timestamp:float'])).tm_wday
#         if (row['item_id:token'],day) in count.keys():
#             count[(row['item_id:token'],day)]+=1
#         else:
#             count[(row['item_id:token'], day)] = 1
#         if count[(row['item_id:token'],day)]>max_count:
#             max_count=count[(row['item_id:token'],day)]
#         # if index % 1000 != 0 and index!=full_data.index[-1]:
#         #     continue
#         # print(index)
#         # x.append(row['item_id:token'])
#         # y.append(day)
#     bias_data = bias_data.sort_values(by='item_id:token')
#     st=int(len(bias_data)/100)
#     for index, row in bias_data.iterrows():
#         day = tm.localtime(int(row['timestamp:float'])).tm_wday
#         # if (row['item_id:token'], day) in count.keys():
#         #     count[(row['item_id:token'], day)] += 1
#         # else:
#         #     count[(row['item_id:token'], day)] = 1
#         # if count[(row['item_id:token'], day)] > max_count:
#         #     max_count = count[(row['item_id:token'], day)]
#         if index % st != 0 :
#             continue
#         x.append(row['item_id:token'])
#         y.append(day)
#     print('count\n', count)
#     print("max_count\n",max_count)
#     print(x[-1],y[-1])
#     #
#     # x=x[:100]
#     # y=y[:100]
#     x.append(1682)
#     y.append(6)
#     # x = np.random.rand(N)  # 包含10个均匀分布的随机值的横坐标数组，大小[0, 1]
#     # y = np.random.rand(N)  # 包含10个均匀分布的随机值的纵坐标数组
#     plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
#     plt.show()
#
#     # id, num = np.unique(count.values, return_counts=True)
#     # print('频数', id)
#     # print('次数', num)
#
#     for k in count.keys():
#         count[k]=max_count/count[k]
#     #print('relative_count\n', count)
#
#     #popularity = count[bias_data['item_id:token']]
#     sampling_weight=[]
#     for index, row in bias_data.iterrows():
#         day=tm.localtime(int(row['timestamp:float'])).tm_wday
#         if  (row['item_id:token'],day) in count.keys():
#             sampling_weight.append(count[(row['item_id:token'],day)])
#         else:
#             sampling_weight.append(0)
#     #sampling_weight = relative_count[bias_data['item_id:token']]
#     # print('popularity', popularity)
#     # print('sampling_weight', sampling_weight)
#
#     #popularity = popularity.values
#     #sampling_weight = sampling_weight.values
#     #bias_data.insert(bias_data.shape[1], 'popularity', popularity)
#     bias_data.insert(bias_data.shape[1], 'sampling_weight', sampling_weight)
#     print('bias_data\n', bias_data)
#
#     data_samples = bias_data.sample(frac=0.5, random_state=2022, weights='sampling_weight')
#
#     data_samples = data_samples.sort_values(by='item_id:token')
#
#     print('data_samples\n', data_samples)
#     count = {}
#     max_count = 0
#     x = []
#     y = []
#     st = int(len(data_samples) / 100)
#     for index, row in data_samples.iterrows():
#
#         day = tm.localtime(int(row['timestamp:float'])).tm_wday
#         if (row['item_id:token'], day) in count.keys():
#             count[(row['item_id:token'], day)] += 1
#         else:
#             count[(row['item_id:token'], day)] = 1
#         if count[(row['item_id:token'], day)] > max_count:
#             max_count = count[(row['item_id:token'], day)]
#         if index % st != 0:
#             continue
#         x.append(row['item_id:token'])
#         y.append(day)
#     print("max_debias_count\n",max_count)
#     #
#     # x = x[:100]
#     # y = y[:100]
#     x.append(1682)
#     y.append(6)
#     plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
#     plt.show()
#     # # id, num = np.unique(bias_data['popularity'], return_counts=True)
#     # # print('id', id)
#     # # print('num', num)
#     # #
#     # # id, num = np.unique(data_samples['popularity'], return_counts=True)
#     # # # print('id', id)
#     # # print('num', num)
#     # data_samples.to_csv("ml_unbias.csv", header=True, index=False, sep='\t',
#     #                     columns=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'])


def split_unbias_TO2(unbias_path=''):  # 将unbias_data划分为valid set和test set  1:1  (key)
    validset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'))
    testset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'))
    data = pd.read_csv(unbias_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    # min = 10000
    for u in user:
        u_data = data[data['user_id:token'].isin([u])]
        # if (len(u_data) < min):
        #    min = len(u_data)
        # print('min', min)
        if (len(u_data) < 2):
            testset = testset.append(u_data)
            continue
        u_valid_data, u_test_data = model_selection.train_test_split(u_data, test_size=0.5, shuffle=True)
        validset = validset.append(u_valid_data)
        testset = testset.append(u_test_data)
    print('dataset_num:', len(data))
    print('validset_num:', len(validset))
    print('testset_num:', len(testset))

    validset.to_csv("ml_valid_unbias.csv", header=True, index=False, sep='\t')
    testset.to_csv("ml_test_unbias.csv", header=True, index=False, sep='\t')



def test_distribution_week(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)
    
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token','timestamp:float', 'watch_ratio:float','intervene_mask:token' ], sep='\t')
    print('data', data)
    # for index, row in data.iterrows():
    #     try:
    #         time = tm.localtime(int(row['timestamp:float'])).tm_wday
    #         # time_origin.append(time)
    #         # cnt[time] += 1
    #     except:
    #         print(row)
    user = data['user_id:token'].unique()
    x = ['>95', '95~90', '90~85', '85~80', '80~75', '<75']
    y1=[0,0,0,0,0,0]
    y2 = [0, 0, 0, 0, 0,0]
    tot=0
    for u in user:
        time_origin=[]
        u_data = data[data['user_id:token'].isin([u])]
        cnt=[0,0,0,0,0,0,0]
        for index, row in u_data.iterrows():
            try:
                time=tm.localtime(int(row['timestamp:float'])).tm_wday
                time_origin.append(time)
                cnt[time]+=1
            except:
                tot+=1
        cnt=sorted(cnt,reverse=True)
        r=cnt[0]/len(u_data)
        if r>=0.95:
            y1[0]+=1
        elif r>=0.9:
            y1[1]+=1
        elif r>=0.85:
            y1[2]+=1
        elif r>=0.80:
            y1[3]+=1
        elif r>=0.75:
            y1[4]+=1
        else:
            y1[5]+=1
        r=(cnt[0]+cnt[1])/len(u_data)
        if r>=0.95:
            y2[0]+=1
        elif r>=0.9:
            y2[1]+=1
        elif r>=0.85:
            y2[2]+=1
        elif r>=0.80:
            y2[3]+=1
        elif r>=0.75:
            y2[4]+=1
        else:
            y2[5]+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    print("totnan",tot)
    print(len(user))
    print(y1)
    y1=[x/len(user)*100 for x in y1]
    #y1=y1/len(user)
    bar1=plt.bar(x, y1, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.bar_label(bar1,fmt='%.2g%%', label_type='edge')
    plt.show()
    print(y2)
    y2 = [x / len(user)*100 for x in y2]
    bar2=plt.bar(x, y2, alpha=0.6) 
    plt.bar_label(bar2,fmt='%.2g%%', label_type='edge')# 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()


def test_distribution_day(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
    #                           'intervene_mask:token'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    x = ['>95', '95~90', '90~85', '85~80', '80~75', '<75']
    y1 = [0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0]
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        cnt = [0 for i in range(24)]
        for index, row in u_data.iterrows():
            time = tm.localtime(int(row['timestamp:float'])).tm_hour
            time_origin.append(time)
            cnt[time] += 1
        cnt = sorted(cnt, reverse=True)
        r = cnt[0] / len(u_data)
        if r >= 0.95:
            y1[0] += 1
        elif r >= 0.9:
            y1[1] += 1
        elif r >= 0.85:
            y1[2] += 1
        elif r >= 0.80:
            y1[3] += 1
        elif r >= 0.75:
            y1[4] += 1
        else:
            y1[5] += 1
        r = (cnt[0] + cnt[1]) / len(u_data)
        if r >= 0.95:
            y2[0] += 1
        elif r >= 0.9:
            y2[1] += 1
        elif r >= 0.85:
            y2[2] += 1
        elif r >= 0.80:
            y2[3] += 1
        elif r >= 0.75:
            y2[4] += 1
        else:
            y2[5] += 1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    print(len(user))
    print(y1)
    y1 = [x / len(user) * 100 for x in y1]
    # y1=y1/len(user)
    bar1 = plt.bar(x, y1, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.bar_label(bar1, fmt='%.2g%%', label_type='edge')
    plt.show()
    print(y2)
    y2 = [x / len(user) * 100 for x in y2]
    bar2 = plt.bar(x, y2, alpha=0.6)
    plt.bar_label(bar2, fmt='%.2g%%', label_type='edge')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()

def test_distribution_day_kuai(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                              'intervene_mask:token'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    #x = ['>25', '25~20', '20~15', '15~10', '10~5', '<5']
    x = ['>50', '50~40', '40~30', '30~20', '20~10', '<10']
    rate=[0.5,0.4,0.3,0.2,0.1,0]
    y1 = [0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0]
    tot=0
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        cnt = [0 for i in range(24)]
        for index, row in u_data.iterrows():
            try:
                time = tm.localtime(int(row['timestamp:float'])).tm_hour
                time_origin.append(time)
                cnt[time] += 1
            except:
                tot+=1
        cnt = sorted(cnt, reverse=True)
        r = cnt[0] / len(u_data)
        for i in range(6):
            if r>=rate[i]:
                y1[i]+=1
                break;
        r = (cnt[0] + cnt[1]) / len(u_data)
        for i in range(6):
            if r >= rate[i]:
                y2[i] += 1
                break;

    print(len(user))
    print(y1)
    y1 = [x / len(user) * 100 for x in y1]
    # y1=y1/len(user)
    bar1 = plt.bar(x, y1, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.bar_label(bar1, fmt='%.2g%%', label_type='edge')
    plt.show()
    print(y2)
    y2 = [x / len(user) * 100 for x in y2]
    bar2 = plt.bar(x, y2, alpha=0.6)
    plt.bar_label(bar2, fmt='%.2g%%', label_type='edge')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()


def test_distribution_week_kuai(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                              'intervene_mask:token'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    # x = ['>25', '25~20', '20~15', '15~10', '10~5', '<5']
    # rate=[0.25,0.2,0.15,0.1,0.05,0]
    x = ['>50', '50~40', '40~30', '30~20', '20~10', '<10']
    rate = [0.5, 0.4, 0.3, 0.2, 0.1, 0]
    y1 = [0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0]
    tot=0
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        cnt = [0 for i in range(24)]
        for index, row in u_data.iterrows():
            try:
                time = tm.localtime(int(row['timestamp:float'])).tm_wday
                time_origin.append(time)
                cnt[time] += 1
            except:
                tot+=1
        cnt = sorted(cnt, reverse=True)
        r = cnt[0] / len(u_data)
        for i in range(6):
            if r>=rate[i]:
                y1[i]+=1
                break;
        r = (cnt[0] + cnt[1]) / len(u_data)
        for i in range(6):
            if r >= rate[i]:
                y2[i] += 1
                break;

    print(len(user))
    print(y1)
    y1 = [x / len(user) * 100 for x in y1]
    # y1=y1/len(user)
    bar1 = plt.bar(x, y1, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.bar_label(bar1, fmt='%.2g%%', label_type='edge')
    plt.show()
    print(y2)
    y2 = [x / len(user) * 100 for x in y2]
    bar2 = plt.bar(x, y2, alpha=0.6)
    plt.bar_label(bar2, fmt='%.2g%%', label_type='edge')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()


def chi_squared_test_week(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['session_id:token', 'item_id:token',  'timestamp:float','number of times:float',], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                        names=['user_id:token', 'item_id:token',  'timestamp:float','rating:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
    #                           'intervene_mask:token'], sep='\t')
    data=data.dropna(axis=0, how='any')
    print('data', data)

    user = data['user_id:token'].unique()
    count=np.zeros((len(user),7), dtype=float)
    fe=np.zeros((len(user),7), dtype=float)
    totu=0
    uc={}
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        if u not in uc:
            uc[u]=totu
            totu+=1
        cnt = [0 for i in range(7)]
        for index, row in u_data.iterrows():
            time = tm.localtime(int(row['timestamp:float'])).tm_wday
            time_origin.append(time)
            cnt[time] += 1
            count[uc[u]][time]+=1

    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(len(user)):
        for j in range(7):
            fe[i][j]=rt[i]*ct[j]/len(data)

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/len(data))
    c=math.sqrt(chi_squared/(chi_squared+len(data)))
    v=math.sqrt(chi_squared/(len(data)*6))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(user)-1)*(7-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(len(user)-1)*(7-1))  # Array of expected counts
    print(res)

def chi_squared_test_hour(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
    #                           'intervene_mask:token'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    count=np.zeros((len(user),24), dtype=float)
    fe=np.zeros((len(user),24), dtype=float)
    x = ['>95', '95~90', '90~85', '85~80', '80~75', '<75']
    y1 = [0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0]
    rt=[]
    ct=[]
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        cnt = [0 for i in range(24)]
        for index, row in u_data.iterrows():
            time = tm.localtime(int(row['timestamp:float'])).tm_hour
            time_origin.append(time)
            cnt[time] += 1
            count[u-1][time]+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(len(user)):
        for j in range(24):
            fe[i][j]=rt[i]*ct[j]/len(data)

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/len(data))
    c=math.sqrt(chi_squared/(chi_squared+len(data)))
    v=math.sqrt(chi_squared/(len(data)*(24-1)))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(user)-1)*(24-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(len(user)-1)*(24-1))  # Array of expected counts
    print(res)
def chi_squared_test_week_it(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'rating:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
    #                           'intervene_mask:token'], sep='\t')
    print('data', data)

    item= data['item_id:token'].unique()
    count=np.zeros((len(item),7), dtype=float)
    fe=np.zeros((len(item),7), dtype=float)
    uc={}
    toti=0
    for i in item:
        time_origin = []
        i_data = data[data['item_id:token'].isin([i])]
        cnt = [0 for i in range(7)]
        if i not in uc:
            uc[i]=toti
            toti+=1
        for index, row in i_data.iterrows():
            time = tm.localtime(int(row['timestamp:float'])).tm_wday
            time_origin.append(time)
            cnt[time] += 1

            count[uc[i]][time]+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(len(item)):
        for j in range(7):
            fe[i][j]=rt[i]*ct[j]/len(data)

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/len(data))
    c=math.sqrt(chi_squared/(chi_squared+len(data)))
    v=math.sqrt(chi_squared/(len(data)*6))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(item)-1)*(7-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(len(item)-1)*(7-1))  # Array of expected counts
    print(res)

def chi_squared_test_hour_it(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
    #                           'intervene_mask:token'], sep='\t')
    print('data', data)

    item = data['item_id:token'].unique()
    count = np.zeros((len(item), 24), dtype=float)
    fe = np.zeros((len(item), 24), dtype=float)
    for i in item:
        time_origin = []
        i_data = data[data['item_id:token'].isin([i])]
        cnt = [0 for i in range(24)]
        for index, row in i_data.iterrows():
            time = tm.localtime(int(row['timestamp:float'])).tm_hour
            time_origin.append(time)
            cnt[time] += 1
            count[i - 1][time] += 1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    rt = count.sum(axis=1)
    ct = count.sum(axis=0)
    for i in range(len(item)):
        for j in range(24):
            fe[i][j] = rt[i] * ct[j] / len(data)

    c = count - fe
    c = (np.power(c, 2) / fe)
    chi_squared = c.sum()
    print(chi_squared)
    phi = math.sqrt(chi_squared / len(data))
    c = math.sqrt(chi_squared / (chi_squared + len(data)))
    v = math.sqrt(chi_squared / (len(data) * 23))
    print(phi, c, v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(item) - 1) * (24 - 1))

    res = stats.chisquare(f_obs=count,  # Array of obversed counts
                          f_exp=fe, axis=None, ddof=(len(item) - 1) * (24 - 1))  # Array of expected counts
    print(res)
def chi_squared_test_week_tr(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float','rating:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
    #                           'intervene_mask:token'], sep='\t')
    print('data', data)
    # sdata=data.sort_values('rating:float')
    # totr=len(data['rating:float'].unique())
    totr=15
    #user = data['user_id:token'].unique()
    count=np.zeros((totr,7), dtype=float)
    fe=np.zeros((totr,7), dtype=float)
    # for u in user:
    #     time_origin = []
    #     u_data = data[data['user_id:token'].isin([u])]
    #     cnt = [0 for i in range(7)]
    #     for index, row in u_data.iterrows():
    #         time = tm.localtime(int(row['timestamp:float'])).tm_wday
    #         time_origin.append(time)
    #         rating=int(row['rating:float'])
    #         cnt[time] += 1
    #         count[rating-1][time]+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    uc={}
    nowr=0
    for index, row in data.iterrows():
        if row['rating:float']>15:
            row['rating:float']=15
        time = tm.localtime(int(row['timestamp:float'])).tm_wday
        count[int(row['rating:float'])-1][time]+=1
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(totr):
        for j in range(7):
            fe[i][j]=rt[i]*ct[j]/len(data)

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/len(data))
    c=math.sqrt(chi_squared/(chi_squared+len(data)))
    v=math.sqrt(chi_squared/(len(data)*(7-1)))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(totr-1)*(7-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(totr-1)*(7-1))  # Array of expected counts
    print(res)

def chi_squared_test_hour_tr(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
    #                           'intervene_mask:token'], sep='\t')
    print('data', data)
    totr=5
    user = data['user_id:token'].unique()
    count=np.zeros((totr,24), dtype=float)
    fe=np.zeros((totr,24), dtype=float)
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        cnt = [0 for i in range(24)]
        for index, row in u_data.iterrows():
            time = tm.localtime(int(row['timestamp:float'])).tm_hour
            time_origin.append(time)
            rating = int(row['rating:float'])
            cnt[time] += 1
            count[rating-1][time]+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(totr):
        for j in range(24):
            fe[i][j]=rt[i]*ct[j]/len(data)

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/len(data))
    c=math.sqrt(chi_squared/(chi_squared+len(data)))
    v=math.sqrt(chi_squared/(len(data)*(totr-1)))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(totr-1)*(24-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(totr-1)*(24-1))  # Array of expected counts
    print(res)


def chi_squared_test_week_kuai(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)


    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                              'intervene_mask:token'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    count=np.zeros((len(user),7), dtype=float)
    fe=np.zeros((len(user),7), dtype=float)
    x = ['>95', '95~90', '90~85', '85~80', '80~75', '<75']
    y1 = [0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0]
    rt=[]
    ct=[]
    tote=0
    uc={}
    totu=0
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        cnt = [0 for i in range(7)]
        if u not in uc:
            uc[u] = totu
            totu+=1
        for index, row in u_data.iterrows():
            try:
                time = tm.localtime(int(row['timestamp:float'])).tm_wday
                time_origin.append(time)
                cnt[time] += 1

                count[uc[u]][time] += 1
            except:
                tote+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    totu=len(uc.keys())
    totn=len(data)-tote
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(len(user)):
        for j in range(7):
            fe[i][j]=rt[i]*ct[j]/totn

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/totn)
    c=math.sqrt(chi_squared/(chi_squared+totn))
    v=math.sqrt(chi_squared/(totn*6))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(user)-1)*(7-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(len(user)-1)*(7-1))  # Array of expected counts
    print(res)

def chi_squared_test_hour_kuai(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                              'intervene_mask:token'], sep='\t')
    print('data', data)

    user = data['user_id:token'].unique()
    count=np.zeros((len(user),24), dtype=float)
    fe=np.zeros((len(user),24), dtype=float)
    x = ['>95', '95~90', '90~85', '85~80', '80~75', '<75']
    y1 = [0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0]
    rt=[]
    ct=[]
    tote=0
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        cnt = [0 for i in range(24)]
        for index, row in u_data.iterrows():
            # time = tm.localtime(int(row['timestamp:float'])).tm_hour
            # time_origin.append(time)
            # cnt[time] += 1
            # count[u-1][time]+=1
            try:
                time = tm.localtime(int(row['timestamp:float'])).tm_wday
                time_origin.append(time)
                cnt[time] += 1
                count[u - 1][time] += 1
            except:
                tote+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    totn = len(data) - tote
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(len(user)):
        for j in range(24):
            fe[i][j]=rt[i]*ct[j]/totn

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/totn)
    c=math.sqrt(chi_squared/(chi_squared+totn))
    v=math.sqrt(chi_squared/(totn*(24-1)))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(user)-1)*(24-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(len(user)-1)*(24-1))  # Array of expected counts
    print(res)

def chi_squared_test_week_kuai_tr(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)


    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                              'intervene_mask:token'], sep='\t')
    print('data', data)
    totr=9
    #ratio=data['watch_ratio:float'].max()
    data = data.dropna(axis=0, how='any')

    user = data['user_id:token'].unique()
    count=np.zeros((totr,7), dtype=float)
    fe=np.zeros((totr,7), dtype=float)
    x = ['>95', '95~90', '90~85', '85~80', '80~75', '<75']
    y1 = [0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0]
    rt=[]
    ct=[]
    tote=0
    uc={}
    totu=0
    for index, row in data.iterrows():
        if row['watch_ratio:float']>10:
            row['watch_ratio:float']=10
        time = tm.localtime(int(row['timestamp:float'])).tm_wday
        count[int(row['watch_ratio:float'])-2][time]+=1
    # for u in user:
    #     time_origin = []
    #     u_data = data[data['user_id:token'].isin([u])]
    #     cnt = [0 for i in range(7)]
    #     if u not in uc:
    #         uc[u] = totu
    #         totu+=1
    #     for index, row in u_data.iterrows():
    #         try:
    #             time = tm.localtime(int(row['timestamp:float'])).tm_wday
    #             time_origin.append(time)
    #             cnt[time] += 1
    #
    #             count[uc[u]][time] += 1
    #         except:
    #             tote+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()

    totn=len(data)
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(totr):
        for j in range(7):
            fe[i][j]=rt[i]*ct[j]/totn

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/totn)
    c=math.sqrt(chi_squared/(chi_squared+totn))
    v=math.sqrt(chi_squared/(totn*6))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(totr-1)*(7-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(totr-1)*(7-1))  # Array of expected counts
    print(res)

def chi_squared_test_week_kuai_it(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)
    #
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                              'intervene_mask:token'], sep='\t')
    data = data.dropna(axis=0, how='any')
    print('data', data)

    item = data['item_id:token'].unique()
    count = np.zeros((len(item), 7), dtype=float)
    fe = np.zeros((len(item), 7), dtype=float)
    for i in range(len(item)):
        time_origin = []
        i_data = data[data['item_id:token'].isin([item[i]])]
        cnt = [0 for i in range(24)]
        for index, row in i_data.iterrows():
            time = tm.localtime(int(row['timestamp:float'])).tm_wday
            time_origin.append(time)
            cnt[time] += 1
            count[i ][time] += 1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    rt = count.sum(axis=1)
    ct = count.sum(axis=0)
    for i in range(len(item)):
        for j in range(7):
            fe[i][j] = rt[i] * ct[j] / len(data)

    c = count - fe
    c = (np.power(c, 2) / fe)
    chi_squared = c.sum()
    print(chi_squared)
    phi = math.sqrt(chi_squared / len(data))
    c = math.sqrt(chi_squared / (chi_squared + len(data)))
    v = math.sqrt(chi_squared / (len(data) * 6))
    print(phi, c, v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(item) - 1) * (7 - 1))

    res = stats.chisquare(f_obs=count,  # Array of obversed counts
                          f_exp=fe, axis=None, ddof=(len(item) - 1) * (7 - 1))  # Array of expected counts
    print(res)

def chi_squared_test_week_ur(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)

    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['session_id:token', 'item_id:token',  'timestamp:float','number of times:float',], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                        names=['user_id:token', 'item_id:token','rating:float','timestamp:float'], sep='\t')
    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
    #                           'intervene_mask:token'], sep='\t')
    data=data.dropna(axis=0, how='any')
    print('data', data)

    user = data['user_id:token'].unique()
    totr=5
    count=np.zeros((len(user),totr), dtype=float)
    fe=np.zeros((len(user),totr), dtype=float)
    totu=0
    uc={}
    # for u in user:
    #     time_origin = []
    #     u_data = data[data['user_id:token'].isin([u])]
    #     if u not in uc:
    #         uc[u]=totu
    #         totu+=1
    #     for index, row in u_data.iterrows():
    #         # time = tm.localtime(int(row['timestamp:float'])).tm_wday
    #         # time_origin.append(time)
    #         #cnt[time] += 1
    #         # if row['rating:float'] > 15:
    #         #     row['rating:float'] = 15
    #         count[uc[u]][int(row['rating:float'])-1]+=1

    for index, row in data.iterrows():
    # time = tm.localtime(int(row['timestamp:float'])).tm_wday
            # time_origin.append(time)
            #cnt[time] += 1
            # if row['rating:float'] > 15:
            #     row['rating:float'] = 15
        u=  row['user_id:token']
        if u not in uc:
            uc[u]=totu
            totu+=1
        count[uc[u]][int(row['rating:float'])-1]+=1
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(len(user)):
        for j in range(totr):
            fe[i][j]=rt[i]*ct[j]/len(data)

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/len(data))
    c=math.sqrt(chi_squared/(chi_squared+len(data)))
    v=math.sqrt(chi_squared/(len(data)*(totr-1)))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(user)-1)*(5-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(len(user)-1)*(5-1))  # Array of expected counts
    print(res)


def chi_squared_test_week_kuai_ur(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)


    # data = pd.read_csv(fulldata_path, header=0,
    #                    names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'timestamp:float', 'watch_ratio:float',
                              'intervene_mask:token'], sep='\t')
    data = data.dropna(axis=0, how='any')
    print('data', data)
    totr=9
    user = data['user_id:token'].unique()
    count=np.zeros((len(user),totr), dtype=float)
    fe=np.zeros((len(user),totr), dtype=float)
    x = ['>95', '95~90', '90~85', '85~80', '80~75', '<75']
    y1 = [0, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 0]
    rt=[]
    ct=[]
    tote=0
    uc={}
    totu=0
    for u in user:
        time_origin = []
        u_data = data[data['user_id:token'].isin([u])]
        cnt = [0 for i in range(7)]
        if u not in uc:
            uc[u] = totu
            totu+=1
        for index, row in u_data.iterrows():
            try:

                if row['watch_ratio:float'] > 10:
                    row['watch_ratio:float'] = 10
                count[uc[u]][int(row['watch_ratio:float']) - 2] += 1
            except:
                tote+=1
        # time_origin = time_origin.long()
        # plt.bar([0,1,2,3,4,5,6], cnt, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        # plt.show()
    totu=len(uc.keys())
    totn=len(data)-tote
    rt=count.sum(axis=1)
    ct=count.sum(axis=0)
    for i in range(len(user)):
        for j in range(totr):
            fe[i][j]=rt[i]*ct[j]/totn

    c=count-fe
    c=(np.power(c,2)/fe)
    chi_squared=c.sum()
    print(chi_squared)
    phi=math.sqrt(chi_squared/totn)
    c=math.sqrt(chi_squared/(chi_squared+totn))
    v=math.sqrt(chi_squared/(totn*(totr-1)))
    print(phi,c,v)
    P_value = 1 - stats.chi2.cdf(x=chi_squared, df=(len(user)-1)*(totr-1))

    res=stats.chisquare(f_obs=count,  # Array of obversed counts
                    f_exp=fe,axis=None,ddof=(len(user)-1)*(totr-1))  # Array of expected counts
    print(res)


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

        return new_tb.sum() ,new_tb.sum() / (total_sum * (min_row_column - 1))


if __name__ == '__main__':
    #transform_df(path='ml_unbias.csv')
    # split_unbiasedset(path='kuai_unbias.csv', sep='\t')
    # positive_filter('kuai_unbias.csv')
    #split_unbiasedset_TO3('ml_unbias_mask.csv', 'ml_train_bias_mask.csv')
    # sampling_weight('ml_data/ml-100k_origin.inter')
    #split_fulldata_TO2('ml-100k_origin.inter')
    #test_distribution_week('ml-100k_origin.inter')
    # test_distribution_day('ml-100k_origin.inter')
    #chi_squared_test_week_it('dataset/amazon_video_games/amazon_video_games.inter')
    chi_squared_test_week_ur('../dataset/amazon_video_games/amazon_video_games.inter')
    chi_squared_test_week('../dataset/netflix/netflix.inter')
    #chi_squared_test_week_tr('dataset/diginetica/diginetica.inter')
    # chi_squared_test_hour_it('dataset/netflix/netflix.inter')
    # chi_squared_test_hour('dataset/netflix/netflix.inter')
    #chi_squared_test_week_ur('dataset/netflix/netflix.inter')
    #chi_squared_test_week_ur('dataset/diginetica/diginetica.inter')
    #chi_squared_test_week_ur('ml-100k_origin.inter')
    #chi_squared_test_week_tr('ml-100k_origin.inter')
    # chi_squared_test_hour_tr('ml-100k_origin.inter')
    #chi_squared_test_week_kuai_it('dataset/kuai/kuai.train.inter')
    #chi_squared_test_hour_kuai('dataset/kuai/kuai.train.inter')
    #chi_squared_test_week_kuai_ur('dataset/kuai/kuai.train.inter')
    #test_distribution_week('dataset/kuai/kuai.train.inter')
    #test_distribution_day_kuai('dataset/kuai/kuai.train.inter')
    #test_distribution_week_kuai('dataset/kuai/kuai.train.inter')
    #sampling_from_biasData(full_path='ml-100k_origin.inter', bias_path='ml_test_bias.csv')
    #split_unbias_TO2(unbias_path='ml_unbias.csv')
    #judge_userID(train_path='ml_train_1.csv',test_path='ml_unbias_1.csv')
