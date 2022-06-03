# Developer Prince - Concrete Compressive Strength Regression Machine Learning Demo
""" This is Regression problem which make use of Data set with 8 Features in order to Determine Concrete Compressive Strength(MPa, Mega Pascals) """

from __future__ import print_function, absolute_import, division
import pandas as pd
import CCST_ML.generate_model as gm
import os
os.getcwd()
os.listdir(os.getcwd())

data = pd.read_csv('data/ccs_data.csv', low_memory=False)
data_label = pd.read_csv('data/ccs_labels.csv', low_memory=False)
FEATURES = ['Cement', 'Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']

labels = pd.DataFrame(data_label)
data_df = pd.DataFrame(data)

data_len = len(data_df.index)
train_labels_len = int(data_len * 0.8)
test_labels_len = int(data_len * 0.2)
train_data_len = int(data_len * 0.8)
test_data_len = int(data_len * 0.2)

train_data =  data_df.iloc[0:train_data_len,:]
test_data = data_df.iloc[0:test_data_len,:]
train_labels  = labels.iloc[0:train_labels_len,:]
test_labels = labels.iloc[0:test_labels_len,:]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data)
print(train_labels)
print(test_data)
print(test_labels)

models = gm.GENERATE_MODEL()
models.__generate__(train_labels=train_labels, train_data=train_data, test_labels=test_labels, test_data=test_data)
