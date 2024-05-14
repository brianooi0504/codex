#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:00:56 2024

@author: brianooi
"""

import numpy as np
import pandas as pd 
from sklearn import datasets
import csv

k = 3 # number of clients

diabetes_X,diabetes_y = datasets.load_diabetes(return_X_y = True)
data = np.hstack((diabetes_X, diabetes_y.reshape(-1,1)))
data = np.delete(data, 1, 1) 
data = np.hstack((np.ones((data.shape[0], 1), dtype=data.dtype), data))

data_test = data[:40, :]
data_train = data[40:, :]
data_train = np.array_split(data_train, k)

df = pd.DataFrame(data_test)
# df.to_csv("data_test.csv", header=False, index=False)

df = pd.DataFrame(data_train[0])
df.to_csv("data_train0.csv", header=False, index=False)

df = pd.DataFrame(data_train[1])
df.to_csv("data_train1.csv", header=False, index=False)

df = pd.DataFrame(data_train[2])
df.to_csv("data_train2.csv", header=False, index=False)

with open('data_test.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_test = np.array(data, dtype=float)