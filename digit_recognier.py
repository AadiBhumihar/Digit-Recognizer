#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:19:45 2017

@author: bhumihar
"""

import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


train_df = pd.read_csv('train.csv')
train_m = train_df.as_matrix()
X = train_m[:5000,1:]
y = train_m[:5000,0]

logistic = LogisticRegression()
logistic.fit(X,y)

test_df = pd.read_csv('test.csv')
test_m = test_df.as_matrix()
t_X = test_m[:,:]

ids = np.arange(len(t_X[:,1]))

label = logistic.predict(t_X)
predictions_file = open("imagePredict.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids, label))
predictions_file.close()
print ('Done.')