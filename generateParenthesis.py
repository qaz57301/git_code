from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
import os
import tqdm
from tqdm import tqdm_notebook
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


path = 'E:/cv/data/binary/'
train = pd.read_csv(path+'train_set.csv')
test = pd.read_csv(path+'test_set.csv')
# print(train.corr())
# print(train.job.value_counts())
test['y']=-1
data = train.append(test).reset_index(drop=True)
cat_col = [i for i in data.select_dtypes(object).columns if i not in ['ID','y']]
for i in tqdm_notebook(cat_col):
    lbl = LabelEncoder()
    data['count_'+i] = data.groupby([i])[i].transform('count')
    data[i] = lbl.fit_transform(data[i].astype(str))

