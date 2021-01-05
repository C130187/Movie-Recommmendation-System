from surprise import Dataset, SVD, Reader,NMF
from surprise.model_selection import cross_validate,train_test_split
from surprise.model_selection import GridSearchCV,RandomizedSearchCV
from surprise import accuracy

import sys
import seaborn as sns
from timeit import Timer
from datetime import datetime
from surprise import Dataset, SVD, Reader,NMF
from surprise.model_selection import cross_validate,train_test_split
from surprise.model_selection import GridSearchCV,RandomizedSearchCV
from surprise import accuracy
from collections import defaultdict
import random
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import warnings
from surprise.prediction_algorithms.baseline_only import BaselineOnly
warnings.filterwarnings("ignore")



data = pd.read_csv('/home/ec2-user/data/final_dataset.csv')




# Splitting Main dataset to a train and a test dataset
user_freq=data.groupby(['userId']).size().reset_index(name='counts')
users_lt3=user_freq[user_freq['counts']<3][['userId']]
users_ge3=user_freq[user_freq['counts']>=3][['userId']]
train1 = pd.merge(data, users_lt3, on=['userId'],how='inner')
data1 = pd.merge(data, users_ge3, on=['userId'],how='inner')
random.seed(2)
test=data1.groupby('userId').sample(frac=.3, random_state=2)
test_idx = data1.index.isin(test.index.to_list())
train = train1.append(data1[~test_idx])
train_df = train
test_df = test



# Splitting Train data to a train and validation dataset for Hyperparameter optimization of Annoy (ANN)
user_freq1 = data.groupby(['userId']).size().reset_index(name='counts')

users_lt2=user_freq1[user_freq1['counts']<2][['userId']]
users_ge2=user_freq1[user_freq1['counts']>=2][['userId']]
train2 = pd.merge(train, users_lt2, on=['userId'],how='inner')
data2 = pd.merge(train, users_ge2, on=['userId'],how='inner')
random.seed(2)

validation = data2.groupby('userId').sample(frac=0.2, random_state=5)
validation_idx = data2.index.isin(validation.index.to_list())
train_data = train2.append(data2[~validation_idx])


reader = Reader(rating_scale=(1,5))
train = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader=reader)
test = Dataset.load_from_df(test[['userId', 'movieId', 'rating']], reader=reader)
raw_ratings = test.raw_ratings
threshold = int(1 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]

test = test.construct_testset(A_raw_ratings)


raw_ratings1 = train.raw_ratings
threshold = int(1 * len(raw_ratings1))
B_raw_ratings = raw_ratings1[:threshold]
train_test = train.construct_testset(B_raw_ratings)



base=BaselineOnly()
trainset = train.build_full_trainset()
base.fit(trainset)


# In[10]:


def get_pred_data(model,test):
    predictions = model.test(test)
    iid=list(map(lambda x: x.iid, predictions))
    uid=list(map(lambda x: x.uid, predictions))
    est=list(map(lambda x: x.est, predictions))
    r_ui=list(map(lambda x: x.r_ui, predictions))
    data = pd.DataFrame({'userId': uid, 'movieId': iid,'predictions':est,'rating':r_ui})
    return(data)

def get_test_data(user,item):
  user = [user] * len(item)
  r_ui = [1]*len(item)
  data = pd.DataFrame({'userId': user, 'movieId': item,'rating':r_ui})
  test = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader=reader)
  raw_ratings = test.raw_ratings
  threshold = int(1 * len(raw_ratings))
  A_raw_ratings = raw_ratings[:threshold]
  test = test.construct_testset(A_raw_ratings)
  return(test)



def df_crossjoin(test, **kwargs):

    df1=pd.DataFrame(test['userId'].unique())
    df2=pd.DataFrame(test['movieId'].unique())
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)
    res.columns=['userId','movieId']
    res['rating']=3.5
    return res

def test_to_surprise(test_all):
    reader = Reader(rating_scale=(1,5))

    test_all = Dataset.load_from_df(test_all[['userId', 'movieId','rating']], reader=reader)

    raw_ratings = test_all.raw_ratings

    threshold = int(1 * len(raw_ratings))
    A_raw_ratings = raw_ratings[:threshold]

    test_all = test_all.construct_testset(A_raw_ratings)
    return(test_all)



train_all=df_crossjoin(train_df)
train_all=test_to_surprise(train_all)




all_predictions=get_pred_data(base,train_all)
all_predictions.drop('rating',axis=1,inplace=True)




merged = pd.merge(train, all_predictions, on=["userId", "movieId"], how="outer")
all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)



all_predictions.to_csv('/home/ec2-user/data/all_predictions_bias_baseline.csv',index=False)






