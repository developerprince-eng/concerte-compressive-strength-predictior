""" This class if for generating a Dataset from the given path """
from __future__ import print_function, absolute_import, division

import os
import pandas as pd
os.getcwd()
os.listdir(os.getcwd())

class resc_data:
    def __init__(self):
        self.is_data = True

    def __read_csv__(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        return (data_set)


    def __obtain_training_data__(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        train_data_len = int(data_length * 0.8)
        train_data = df.iloc[0:train_data_len,:]
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)

        train_data = (train_data - mean) / std
        return(train_data)


    def __obtain_testing_data__(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        test_data_len = int(data_length * 0.2)
        test_data = df.iloc[0:test_data_len,:]
        mean = test_data.mean(axis=0)
        std = test_data.std(axis=0)

        test_data = (test_data - mean) / std
        return(test_data)



