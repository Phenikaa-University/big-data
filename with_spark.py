#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:09:29 2023

@author: cngvng
"""

import time

# from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

from pyspark.mllib.classification import  LogisticRegressionWithLBFGS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.linear_model import LogisticRegression
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


conf = SparkConf()
conf.setMaster('local')
conf.setAppName('spark-basic')
sc = SparkContext(conf=conf)
data_file = 'data500K_10c.csv'


df= sc.textFile("data/"+ data_file).map(lambda line: line.split(","))
dataset  = df.map(lambda x: LabeledPoint(x[0], x[1:]))

acc_with_spark = list()
time_with_spark = []
for i in range(10):
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3])
    start = time.time()
    # # Decision Tree model
    # model = DecisionTree.trainClassifier(trainingData,  numClasses=10, categoricalFeaturesInfo={}, impurity='gini')
    # Naive Bayes model
    model = NaiveBayes.train(trainingData, 1.0)
    # # Logistic Regression model
    # model = LogisticRegressionWithLBFGS.train(trainingData)
    
    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))    
    
    end = time.time() -start
    print ('time : ', end)
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    
    acc =0
    acc = (labelsAndPredictions.filter( lambda lp: lp[0] == lp[1]).count()) / float(testData.count())
    acc_with_spark.append(acc*100)
    time_with_spark.append(end)

print('Test Error  with spark = ' , sum(acc_with_spark) / len(acc_with_spark))
print('Time with spark: ', sum(time_with_spark) / len(time_with_spark))

acc_without_spark = list()
time_without_spark = []

df = pd.read_csv('data/' + data_file ,header= None)    
r,c = df.shape


df.rename(columns = {0:'class'}, inplace = True)

Y = df[['class']]  
X = df.iloc[:,df.columns !='class']
for i in range(10):

    # df.columns.values[0] = 'class'   # thay ten bang class
  

    # split data
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,  stratify=Y, train_size=0.7, random_state=100000)
    
            
    start = time.time()
    # # decision tree     
    # Naive bayes
    model = GaussianNB()
    # # Logistic Regression 
    # model = LogisticRegression()
    model.fit(X_Train, Y_Train)            
    predictions = model.predict(X_Test)
    end = time.time() -start
    print(end)
    acc_without_spark.append(metrics.accuracy_score(Y_Test, predictions)*100)
    time_without_spark.append(end)
print("Accuracy without spark: ",sum(acc_without_spark) / len(acc_without_spark))
print("Time: without spark: ", sum(time_without_spark) / len(time_without_spark))

# Tạo các điểm trên trục x tương ứng với từng lần chạy
x = range(1, len(acc_with_spark) + 1)

# time_without_spark = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# acc_without_spark = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Vẽ biểu đồ đường
plt.plot(x, time_with_spark, marker='o', label='Spark')
plt.plot(x, time_without_spark, marker='o', label='Non-Spark')

# Đặt nhãn cho trục x và y
plt.xlabel('Run')
plt.ylabel('Execution Time (s)')

# Đặt tiêu đề cho biểu đồ
plt.title(f'Comparison of {model} Time with '+ data_file)

# Đặt chú thích cho các đường trên biểu đồ
plt.legend()
plt.savefig(f"results/{data_file}_{model}_time.png")
plt.clf()

# Tạo các điểm trên trục x tương ứng với từng lần chạy
x = range(1, len(acc_with_spark) + 1)

# Vẽ biểu đồ đường
plt.plot(x, acc_with_spark, marker='o', label='Spark')
plt.plot(x, acc_without_spark, marker='o', label='Non-Spark')

# Đặt nhãn cho trục x và y
plt.xlabel('Run')
plt.ylabel('Accuracy')

# Đặt tiêu đề cho biểu đồ
plt.title(f'Comparison of {model} Accuracy with '+ data_file)

# Đặt chú thích cho các đường trên biểu đồ
plt.legend()
plt.savefig(f"results/{data_file}_{model}_acc.png")

plt.clf()
resuluts = []
resuluts.append(acc_with_spark)
resuluts.append(acc_without_spark)
names = ('With Spark', 'Without Spark')
plt.title(f'Accuracy Comparison based {model}')
plt.boxplot(resuluts, labels=names)
plt.ylabel('Accuracy')
plt.savefig(f"results/{data_file}_{model}_acc_box_plot.png")
