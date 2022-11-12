import pandas as pd
import numpy as np

# np.set_printoptions(threshold=np.inf)
import torch
from sklearn import model_selection
import time as tm

mx_time=0
mi_time=0
norm_time={}
totb=7
bt=[i/totb for i in range(totb)]

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
    print("train_user_num: ",len(train_data['user_id:token'].value_counts()))
    print("train_item_num: ", len(train_data['item_id:token'].value_counts()))
    train_cnt = {}
    max_train_count = 0
    for index, row in train_data.iterrows():
        # day = tm.localtime(int(row['timestamp:float'])).tm_wday
        # item = row['timestamp:float']
        # term = (item, day)
        nt = norm_time[int(row['timestamp:float'])]
        block = get_block(nt)
        item = row['item_id:token']
        term = (item, block)
        if term not in train_cnt:
            train_cnt[term] = 1
        else:
            train_cnt[term] += 1
        if train_cnt[term] > max_train_count:
            max_train_count = train_cnt[term]
    print("test_max_pop:", max_train_count)


# 以下三个函数用于生成半合成数据集
def split_fulldata_TO2(fulldata_path=''):  # 将dataset(ml-100k_origin)划分为train set和test set  1：1  (key)
    trainset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'))
    testset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'))
    data = pd.read_csv(fulldata_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
    # print('data', data)
    normlize_time(data)
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

def normlize_time(data):
    mx_time=data['timestamp:float'].max()
    mi_time=data['timestamp:float'].min()
    for index, row in data.iterrows():
        t=int(row['timestamp:float'])
        if t not in norm_time:
            norm_time[t]=(t-mi_time)/(mx_time-mi_time)
def get_block(t):
    pos=-1
    for i in range(len(bt)):
        if t<bt[i]:
            pos=i-1
            break;
    if pos==-1:
        pos=totb-1
    return pos
def sampling_from_biasData(full_path='', bias_path=''):  # 从整体数据计算流行度，并基于此，从bias_data中采样50%作为unbias_data  (key)
    full_data = pd.read_csv(full_path, header=0,
                            names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'],
                            sep='\t')
    # print('full_data\n', full_data)
    bias_data = pd.read_csv(bias_path, header=0,
                            names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'],
                            sep='\t')
    # print('bias_data\n', bias_data)

    count=[]
    cnt={}
    max_count=0
    for index, row in full_data.iterrows():

        # day = tm.localtime(int(row['timestamp:float'])).tm_wday
        nt=norm_time[int(row['timestamp:float'])]
        block=get_block(nt)
        item=row['item_id:token']
        term=(item,block)
        if term not in cnt:
            cnt[term]=1
        else:
            cnt[term] += 1
    for index, row in full_data.iterrows():
        # day = tm.localtime(int(row['timestamp:float'])).tm_wday
        # item = row['item_id:token']
        # term = (item, day)
        nt = norm_time[int(row['timestamp:float'])]
        block = get_block(nt)
        item = row['item_id:token']
        term = (item, block)
        if cnt[term]>max_count:
            max_count=cnt[term]
        count.append(cnt[term])
    #count = full_data['item_id:token'].value_counts()
    # print('count\n', count)

    # id, num = np.unique(count.values, return_counts=True)
    # print('频数', id)
    # print('次数', num)

    # max_count = count.max()
#    relative_count = max_count / count  # 相对流行度的倒数
 #   print('relative_count\n', relative_count)
    popularity=[]
    sampling_weight=[]
    for index, row in bias_data.iterrows():
        # day = tm.localtime(int(row['timestamp:float'])).tm_wday
        # item = row['timestamp:float']
        # term = (item, day)
        nt = norm_time[int(row['timestamp:float'])]
        block = get_block(nt)
        item = row['item_id:token']
        term = (item, block)
        popularity.append(cnt[term])
        #popularity = count[bias_data['item_id:token']]
        sampling_weight.append(max_count / cnt[term] )
        #sampling_weight = relative_count[bias_data['item_id:token']]
    # print('popularity', popularity)
    # print('sampling_weight', sampling_weight)

    # popularity = popularity.values
    # sampling_weight = sampling_weight.values
    bias_data.insert(bias_data.shape[1], 'popularity', popularity)
    bias_data.insert(bias_data.shape[1], 'sampling_weight', sampling_weight)
    #print('bias_data\n', bias_data)

    data_samples = bias_data.sample(frac=0.5, random_state=2022, weights='sampling_weight')
    #print('data_samples\n', data_samples)

    id, num = np.unique(bias_data['popularity'], return_counts=True)
    print('id', id)
    print('num', num)

    id, num = np.unique(data_samples['popularity'], return_counts=True)
    # print('id', id)
    print('num', num)
    data_samples.to_csv("ml_unbias.csv", header=True, index=False, sep='\t',
                        columns=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'])
    intervened_train=pd.concat([bias_data, data_samples, data_samples]).drop_duplicates(keep=False)
    #intervened_train.drop(['popularity','sampling_weight'])
    intervened_train.to_csv('ml_train_bias.csv',header=False,mode='a', index=False, sep='\t',columns=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'])

def split_unbias_TO2(unbias_path=''):  # 将unbias_data划分为valid set和test set  1:1  (key)
    validset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'))
    testset = pd.DataFrame(
        columns=('user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'))
    data = pd.read_csv(unbias_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], sep='\t')
   # print('data', data)

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
    count = []
    valid_cnt = {}
    max_valid_count = 0
    for index, row in validset.iterrows():
        # day = tm.localtime(int(row['timestamp:float'])).tm_wday
        # item = row['timestamp:float']
        # term = (item, day)
        nt = norm_time[int(row['timestamp:float'])]
        block = get_block(nt)
        item = row['item_id:token']
        term = (item, block)
        if term not in valid_cnt:
            valid_cnt[term] = 1
        else:
            valid_cnt[term] += 1
        if valid_cnt[term]>max_valid_count:
            max_valid_count=valid_cnt[term]
    print("valid_max_pop:",max_valid_count)
    test_cnt = {}
    max_test_count = 0
    for index, row in testset.iterrows():
        # day = tm.localtime(int(row['timestamp:float'])).tm_wday
        # item = row['timestamp:float']
        # term = (item, day)
        nt = norm_time[int(row['timestamp:float'])]
        block = get_block(nt)
        item = row['item_id:token']
        term = (item, block)
        if term not in test_cnt:
            test_cnt[term] = 1
        else:
            test_cnt[term] += 1
        if test_cnt[term]>max_test_count:
            max_test_count=test_cnt[term]
    print("test_max_pop:",max_test_count)

    print("valid_user_num: ", len(validset['user_id:token'].value_counts()))
    print("valid_item_num: ", len(validset['item_id:token'].value_counts()))
    print("test_user_num: ", len(testset['user_id:token'].value_counts()))
    print("test_item_num: ", len(testset['item_id:token'].value_counts()))

    validset.to_csv("ml_valid_unbias.csv", header=True, index=False, sep='\t')
    testset.to_csv("ml_test_unbias.csv", header=True, index=False, sep='\t')





if __name__ == '__main__':
    #transform_df(path='ml_unbias.csv')
    # split_unbiasedset(path='kuai_unbias.csv', sep='\t')
    # positive_filter('kuai_unbias.csv')
    #split_unbiasedset_TO3('ml_unbias_mask.csv', 'ml_train_bias_mask.csv')
    #sampling_weight('ml_data/ml-100k_origin.inter')
    # split_fulldata_TO2('ml-100k.inter')
    # sampling_from_biasData(full_path='ml-100k.inter', bias_path='ml_test_bias.csv')
    # split_unbias_TO2(unbias_path='ml_unbias.csv')
    judge_userID(train_path='ml_train_bias.csv', test_path='ml_test_unbias.csv')

