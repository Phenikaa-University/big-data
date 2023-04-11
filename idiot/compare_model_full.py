"""
Created on Thu Oct 20 10:03:41 2022
@author: cngvng
"""

import pandas as pd
import numpy as np
import time

import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import random



from utils import *
import sys
sys.path.append("/home/cngvng/PKA/big-data/")

n_start = 1
n_end = 5
types = "full"
normalized = False
binary_classify = False

data_path_full = './data/UNSW-full/UNSW-NB15_{}.csv'  # There are 4 input csv files
data_path_feature = './data/UNSW-full/NUSW-NB15_features.csv'

""" Reading full data """

all_data = reading_data_full(data_path_full=data_path_full, data_path_feature=data_path_feature,
                             n_start=n_start, n_end=n_end)

""" Preprocessing data """

df = preprocessing_data_unsw_full(all_data=all_data, binary_classify=binary_classify, normalized=normalized)

""" Visualize data """

visualize_data(df=all_data)


""" Comparison methods """

acc_DT =list()
acc_RF= list()

time_DT_total = list()
time_RF_total = list()

X = df.drop(columns=['label'], axis=1)
y = df['label']

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X,  y, stratify=y, test_size=0.3, random_state = random.randint(0,100000))
    model = DecisionTreeClassifier()
    start_DT = time.process_time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_DT = time.process_time()
    time_DT = end_DT - start_DT
    time_DT_total.append(time_DT)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_DT.append(acc)
    
    rf_model = RandomForestClassifier(max_depth=5, n_estimators=10)
    start_RF = time.process_time()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    end_RF = time.process_time()
    time_RF = end_RF - start_RF
    time_RF_total.append(time_RF)
    acc_R = metrics.accuracy_score(y_test, y_pred_rf)
    acc_RF.append(acc_R)
    
results = []
results.append(acc_DT)
results.append(acc_RF)

names =('Decision tree', 'Random forest')
fig = plt.figure()
fig.suptitle('Algorithm Comparison based Accuracy')
plt.boxplot(results, labels=names)
plt.ylabel('Accuracy') 
plt.savefig("../plots/compare_model/full/Algorithm-comparison-acc-full-50loop-full-multi.pdf")

results_time = []
results_time.append(time_DT_total)
results_time.append(time_RF_total)
names =('Decision tree', 'Random forest')
fig = plt.figure()
fig.suptitle('Algorithm Comparison base Running time')
plt.boxplot(results_time, labels=names)
plt.ylabel('Running time') 
plt.savefig("../plots/compare_model/full/Algorithm-comparison-time-full-50loop-full-multi.pdf")