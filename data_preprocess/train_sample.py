import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
import numpy as np
import math
import pandas as pd
import csv
import seaborn as sns
# 打开 CSV 文件并读取数据
def get_statistic(path):
    data = pd.read_csv(path, header=0,
                            names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'],
                            sep='\t')

    cnt = {}
    max_count = 0
    for index, row in data.iterrows():
        day = int(row['wday:float'])
        item = row['item_id:token']
        term = (item, day)

        if term not in cnt:
            cnt[term] = 1
        else:
            cnt[term] += 1
            if cnt[term]>max_count:
                max_count=cnt[term]


    print(max_count)
    print(len(data))
    probability = {}

    for d, f in cnt.items():
        probability[d] = f / len(data)


    # 计算信息熵
    entropy = 0
    for p in probability.values():
        entropy -= p * math.log(p, 2)

    print(entropy)
    # with open(path+'_statistic.csv', 'w', newline='') as csvfile:
    #     # Create a CSV writer object
    #     writer = csv.DictWriter(csvfile, fieldnames=['key', 'value','percent'])
    #
    #     # Write the headers to the file
    #     writer.writeheader()
    #
    #     # Iterate over the dictionary and write each key-value pair to the file
    #     for key, value in cnt.items():
    #         writer.writerow({'key': key, 'value': value,'percent':value/max_count})


def data_plot(full_path,test_path):
    counts = [0, 0, 0, 0, 0]

    # Open the CSV file for reading
    tot=0
    with open(test_path, 'r') as csvfile:
        # Create a CSV reader object
        reader = csv.DictReader(csvfile)

        # Iterate over the rows of the file
        for row in reader:
            # Extract the value from the row
           # value = float(row['percent'])
            value = int(row['value'])
            num=int(row['value'])
            tot+=num
            # Increment the appropriate count based on the value
            if value < 20:
                counts[0] += num
            elif value < 40:
                counts[1] += num
            elif value < 60:
                counts[2] += num
            elif value < 80:
                counts[3] += num
            else:
                counts[4] += num

    # Print the counts
    print(counts)

    print(tot)
    p=[x/tot for x in counts]
    print(p)
    # Create a figure and an axis
    fig, ax = plt.subplots()
    bar_width = 0.5
    # Plot the lists as bar plots
    x = [i - bar_width/2 for i in range(len(counts))]
    ax.bar(x, p,bar_width, label='List 1')
    counts = [0, 0, 0, 0, 0]

    # Open the CSV file for reading
    tot = 0
    with open(full_path, 'r') as csvfile:
        # Create a CSV reader object
        reader = csv.DictReader(csvfile)

        # Iterate over the rows of the file
        for row in reader:
            # Extract the value from the row
            # value = float(row['percent'])
            value = int(row['value'])
            num = int(row['value'])
            tot += num
            # Increment the appropriate count based on the value
            if value < 20:
                counts[0] += num
            elif value < 40:
                counts[1] += num
            elif value < 60:
                counts[2] += num
            elif value < 80:
                counts[3] += num
            else:
                counts[4] += num

    # Print the counts
    print(counts)
    print(tot)
    p = [x / tot for x in counts]
    print(p)

    x=[i+bar_width/2 for i in range(len(counts))]

    ax.bar(x, p,bar_width, label='List 2')

    # Set the x-axis labels
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(['0~20', '20~40', '40~60', '60~80', '>=80'])

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
def test_split(full_path,test_path):
    data = pd.read_csv(full_path, header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'],
                       sep='\t')

    cnt = {}
    max_count = 0
    for index, row in data.iterrows():
        day = int(row['wday:float'])
        item = row['item_id:token']
        term = (item, day)

        if term not in cnt:
            cnt[term] = 1
        else:
            cnt[term] += 1
            if cnt[term] > max_count:
                max_count = cnt[term]

    sorted_dict = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    p=0
    tot=0
    b=len(cnt.items())/5
    num=[b*(i+1) for i in range(4)]
    num.append(len(cnt.items()))
    idx={}
    #idx=[[] for i in range(5)]
    for item in sorted_dict:

        if tot>num[p]:
            p+=1

        #idx[p].append(k)
        idx[item[0]]=p
        tot+=1
    data = pd.read_csv(test_path+'.inter', header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'],
                       sep='\t')
    ind=[[] for i in range(5)]
    for index, row in data.iterrows():
        day = int(row['wday:float'])
        item = row['item_id:token']
        term = (item, day)
        ind[idx[term]].append(index)

    for i in range(5):
        d=data.iloc[ind[i],:]
        d.to_csv(test_path + "_"+str(i)+".inter", header=True, index=False, sep='\t',
                            columns=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'])

def train_split(path,threshold=None):
    data = pd.read_csv(path + '.inter', header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'],
                       sep='\t')
    #if threshold!=None:
    t_data=data.loc[data['rating:float']>threshold]
    left_data=data.loc[data['rating:float']<=threshold]
    ratio=[(i+1)*0.2 for i in range(5)]
    for i in range(len(ratio)):
        df = t_data.sample(frac=(1-ratio[i]), random_state=2022)
        d=pd.concat([df,left_data])
        d.to_csv(path + "_" + str(i) + ".inter", header=True, index=False, sep='\t',
                 columns=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'])
        rd=data.sample(frac=1-len(t_data)*ratio[i]/len(data), random_state=2022)
        rd.to_csv(path + "_rand_" + str(i) + ".inter", header=True, index=False, sep='\t',
                 columns=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'])

def train_sample(path):
    data = pd.read_csv(path+"_full.csv", header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'],
                       sep='\t')

    cnt = {}
    max_count = 0
    for index, row in data.iterrows():
        day = int(row['wday:float'])
        item = row['item_id:token']
        term = (item, day)

        if term not in cnt:
            cnt[term] = 1
        else:
            cnt[term] += 1
            if cnt[term] > max_count:
                max_count = cnt[term]

    data = pd.read_csv(path + ".train.inter", header=0,
                       names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'],
                       sep='\t')
    ratio=[10]
    for i in range(len(ratio)):
        weight=[]
        for index, row in data.iterrows():
            day = int(row['wday:float'])
            item = row['item_id:token']
            term = (item, day)
            weight.append(pow(cnt[term]/max_count,ratio[i]))
        tdata=data.join(pd.Series(weight, name='weight'))
        sampled_df = tdata.sample(frac=0.5, weights='weight', replace=False)
        sampled_df.to_csv(path + ".train_sample_" + str(ratio[i]) + ".inter", header=True, index=False, sep='\t',
                 columns=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float', 'wday:float'])
    #if threshold!=None:



if __name__ == '__main__':
    path="food_global"
    train_sample(path)
    get_statistic('food_global.train_sample_10.inter')
    # get_statistic('ml_1m_global.train_sample_0.5.inter')
    # get_statistic('ml_1m_global.train_sample_1.inter')
    # get_statistic('ml_1m_global.train_sample_2.inter')
    # get_statistic('food_global.train_sample_0.5.inter')
    # get_statistic('food_global.train_sample_1.inter')
    # get_statistic('food_global.train_sample_2.inter_old')
    # get_statistic('ml_1m_global.test.inter')
    # get_statistic('food_global_old/food_global_old.train.inter')
    # get_statistic('food_global_old/food_global_old.test.inter')
    # get_statistic('Amazon_Digital_Music_global/Amazon_Digital_Music_global.train.inter')
    # get_statistic('Amazon_Digital_Music_global/Amazon_Digital_Music_global.test.inter')
    #data_plot('ml_1m_global_full.csv_statistic.csv','ml_1m_global.test.inter_old_statistic.csv')
    #get_statistic('food_global_full.csv')
    # data_plot('ml_1m_global_full.csv_statistic.csv')