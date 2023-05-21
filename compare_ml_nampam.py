# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:37:46 2023

@author: Admin
"""
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier 
import random
import numpy as np
import math
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
# instantiate labelencoder object
from sklearn.tree import export_text

from IPython.display import Image  
from sklearn.tree import export_graphviz

from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dbfread import dbf
from simpledbf import Dbf5
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import seaborn as sns

def cluster_result():
    ketqua_chay = pd.read_csv('c:\\nampam\\ketqua_chay_cluster.csv' ) 
    
    cols = list(ketqua_chay.columns)
    cols
    X = ketqua_chay[["svm_all_FR", "data_add_d"]]

    kmeans = KMeans(n_clusters=5).fit(X)
    sfr_kmean = kmeans.labels_

    X = ketqua_chay[["logistic_a", "data_add_d"]]
    kmean = KMeans(n_clusters=5).fit(X)
    LR_fr_kmean = kmean.labels_

    ketqua_chay['SVM_FR_K'] = sfr_kmean
    ketqua_chay['LR_FRK1'] = LR_fr_kmean
    
#    ketqua_chay.to_csv('c:\\nampam\\ketqua_cluster.csv', index= False);


# https://www.imranabdullah.com/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot

def plotHistogram():
    print ('thanh')
    data = pd.read_csv('c:\\nampam\\data2\\ketqua_allmap_lr_svm_bys_knn.csv')
    cols = list(data.columns)

    x= data[['svm_all_AHP']]
 
    fig = plt.figure(figsize=(12,10))

    plt.style.use('ggplot')

    plt.grid(False)
    
    plt.xlabel("Landslide susceptibility value", fontsize = 16,fontweight='bold')
    plt.xticks(fontsize=16,fontweight='bold')
    
    plt.ylabel("Frequency", fontsize = 16,fontweight='bold')    

    plt.yticks(np.arange(0, 150000, step=40000))
    plt.yticks(fontsize=14,fontweight='bold')

    
    plt.hist(x, bins=200, ec = 'black')
    plt.show()
    
   

def tinh_VIF():
    print ('ok')    
    

    dataframe = pd.read_csv("C:\\nampam\\training_part_FR.csv")
    dataframe = dataframe[dataframe['landslide'] > 0]
    df = dataframe
    df = pd.read_csv('c:\\nampam\\np_map_FR.csv' ) 
    df = df.iloc[:,3:]
    # df = df.drop(['FR_roadbuffer'], axis=1)
    

    X =df
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
# calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
    print(vif_data)
    
def run_sosanh():

    ketqua_training = pd.read_csv('c:\\nampam\\ketqua_training.csv')

    
    ## Bang de hien thi ket qua
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc']) 
    result_parameter = pd.DataFrame(columns=['classifiers', 'fpr','tpr','acc','MAE','RMSE']) 

### logistic FR       
    # logistic_fr = ketqua_training[['landslide','logistic_training','n_lr_fr']]
    # y_real = ketqua_training[['landslide']]
    
    # arr = logistic_fr[['logistic_training']] 
    
    # new_arr = np.where(arr > 0.5, 1, 0)
 
    # ## Tinh FPR, TPR, AUC:     
    # fpr, tpr, _ = roc_curve(y_real,  new_arr)
    
    
    # auc = roc_auc_score(y_real, new_arr)
    # acc = metrics.accuracy_score(y_real, new_arr)
    
    
    # result_table = result_table.append({'classifiers':'LR_FR',
    #                                     'fpr':fpr, 
    #                                     'tpr':tpr, 
    #                                     'auc':auc}, ignore_index=True)

    # y_true, predictions = np.array(y_real), np.array(arr)
    # mae = np.mean(np.abs(y_true - predictions))
    # mse = mean_squared_error(y_real, arr)
    # rmse = np.sqrt(mse)
      
    # result_parameter= result_parameter.append({'classifiers':'LR_FR','tpr':tpr[1],'fpr':fpr[1],'acc':acc,'MAE':mae,'RMSE':rmse}, ignore_index=True)
       
    
#######          svm      RF
    
    svm_fr = ketqua_training[['landslide','svm_training','n_svm_fr']]
    y_real = ketqua_training[['landslide']]
    
    
    arr = svm_fr[['svm_training']] 
    print (np.median(np.array(arr)))  ## Trung vi 
    print (np.mean(np.array(arr)))  ## Trung vi 

    pivot =  np.mean(np.array(arr))      
    
    pivot = 0.952924
    new_arr = np.where(arr > pivot, 1, 0)
 
    ## Tinh FPR, TPR, AUC:     
    fpr, tpr, _ = roc_curve(y_real,  new_arr)
    auc = roc_auc_score(y_real, new_arr)
    acc = metrics.accuracy_score(y_real, new_arr)
    
    
    result_table = result_table.append({'classifiers':'SVM_FR',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    
    
    y_true, predictions = np.array(y_real), np.array(arr)
    mae = np.mean(np.abs(y_true - predictions))
    mse = mean_squared_error(y_real, arr)
    rmse = np.sqrt(mse)
      
    result_parameter= result_parameter.append({'classifiers':'SVM_RF','tpr':tpr[1],'fpr':fpr[1],'acc':acc,'MAE':mae,'RMSE':rmse}, ignore_index=True)
       
    cols = list(ketqua_training.columns)


###################### Baysian RF

    bayes_fr = ketqua_training[['landslide','baysian_training_RF','n_baysian_rf']]
    y_real = ketqua_training[['landslide']]
    
    
    arr = bayes_fr[['baysian_training_RF']] 
    
    print (np.median(np.array(arr)))  ## Trung vi 
    print (np.mean(np.array(arr)))  ## Trung vi 


    pivot =  np.mean(np.array(arr))      
    pivot = 0.661429
    new_arr = np.where(arr > pivot, 1, 0)
    
 
    ## Tinh FPR, TPR, AUC:     
    fpr, tpr, _ = roc_curve(y_real,  new_arr)
    auc = roc_auc_score(y_real, new_arr)
    acc = metrics.accuracy_score(y_real, new_arr)
    
    
    result_table = result_table.append({'classifiers':'Baysian_FR',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    
    
    y_true, predictions = np.array(y_real), np.array(arr)
    mae = np.mean(np.abs(y_true - predictions))
    mse = mean_squared_error(y_real, arr)
    rmse = np.sqrt(mse)
      
    result_parameter= result_parameter.append({'classifiers':'Baysian_FR','tpr':tpr[1],'fpr':fpr[1],'acc':acc,'MAE':mae,'RMSE':rmse}, ignore_index=True)
       

####################  KNN RF
    

    knn_fr = ketqua_training[['landslide','KNN_training_RF','n_KNN_rf']]
    y_real = ketqua_training[['landslide']]
    
    
    arr = knn_fr[['KNN_training_RF']] 
    
    print (np.median(np.array(arr)))  ## Trung vi 
    print (np.mean(np.array(arr)))  ## Trung vi 

    pivot =  np.mean(np.array(arr))      
    pivot = 0.870588
    new_arr = np.where(arr > pivot, 1, 0)
 
    ## Tinh FPR, TPR, AUC:     
    fpr, tpr, _ = roc_curve(y_real,  new_arr)
    auc = roc_auc_score(y_real, new_arr)
    acc = metrics.accuracy_score(y_real, new_arr)
    
    
    result_table = result_table.append({'classifiers':'KNN_FR',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    
    
    y_true, predictions = np.array(y_real), np.array(arr)
    mae = np.mean(np.abs(y_true - predictions))
    mse = mean_squared_error(y_real, arr)
    rmse = np.sqrt(mse)
      
    result_parameter= result_parameter.append({'classifiers':'KNN_FR','tpr':tpr[1],'fpr':fpr[1],'acc':acc,'MAE':mae,'RMSE':rmse}, ignore_index=True)
       

########################################  AHP


    
 ### logistic ahp       
    # logistic_fr = ketqua_training[['landslide','logistic_training_ahp','n_lr_ahp']]
    # y_real = ketqua_training[['landslide']]
    
    # arr = logistic_fr[['logistic_training_ahp']] 
    
    # new_arr = np.where(arr > 0.5, 1, 0)
 
    # ## Tinh FPR, TPR, AUC:     
    # fpr, tpr, _ = roc_curve(y_real,  new_arr)
    
    
    # auc = roc_auc_score(y_real, new_arr)
    # acc = metrics.accuracy_score(y_real, new_arr)
    
    
    # result_table = result_table.append({'classifiers':'LR_AHP',
    #                                     'fpr':fpr, 
    #                                     'tpr':tpr, 
    #                                     'auc':auc}, ignore_index=True)

    # y_true, predictions = np.array(y_real), np.array(arr)
    # mae = np.mean(np.abs(y_true - predictions))
    # mse = mean_squared_error(y_real, arr)
    # rmse = np.sqrt(mse)
      
    # result_parameter= result_parameter.append({'classifiers':'LR_AHP','tpr':tpr[1],'fpr':fpr[1],'acc':acc,'MAE':mae,'RMSE':rmse}, ignore_index=True)
       
    
#######          svm      AHP
    
    svm_fr = ketqua_training[['landslide','svm_training_ahp','n_svm_ahp']]
    y_real = ketqua_training[['landslide']]
    
    
    arr = svm_fr[['svm_training_ahp']] 
    
    print (np.median(np.array(arr)))  ## Trung vi 
    print (np.mean(np.array(arr)))  ## Trung vi 

    pivot =  np.mean(np.array(arr))      
    pivot = 0.957432
    new_arr = np.where(arr > pivot, 1, 0)
 
    ## Tinh FPR, TPR, AUC:     
    fpr, tpr, _ = roc_curve(y_real,  new_arr)
    auc = roc_auc_score(y_real, new_arr)
    acc = metrics.accuracy_score(y_real, new_arr)
    
    
    result_table = result_table.append({'classifiers':'SVM_AHP',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    
    
    y_true, predictions = np.array(y_real), np.array(arr)
    mae = np.mean(np.abs(y_true - predictions))
    mse = mean_squared_error(y_real, arr)
    rmse = np.sqrt(mse)
      
    result_parameter= result_parameter.append({'classifiers':'SVM_AHP','tpr':tpr[1],'fpr':fpr[1],'acc':acc,'MAE':mae,'RMSE':rmse}, ignore_index=True)
       
    cols = list(ketqua_training.columns)


###################### Baysian RF

    bayes_fr = ketqua_training[['landslide','baysian_training_AHP','n_baysian_AHP']]
    y_real = ketqua_training[['landslide']]
    
    
    arr = bayes_fr[['baysian_training_AHP']] 
    
    print (np.median(np.array(arr)))  ## Trung vi 
    print (np.mean(np.array(arr)))  ## Trung vi 

    pivot =  np.mean(np.array(arr))      
    pivot = 0.99607
    new_arr = np.where(arr > pivot, 1, 0)
 
    ## Tinh FPR, TPR, AUC:     
    fpr, tpr, _ = roc_curve(y_real,  new_arr)
    auc = roc_auc_score(y_real, new_arr)
    acc = metrics.accuracy_score(y_real, new_arr)
    
    
    result_table = result_table.append({'classifiers':'Baysian_AHP',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    
    
    y_true, predictions = np.array(y_real), np.array(arr)
    mae = np.mean(np.abs(y_true - predictions))
    mse = mean_squared_error(y_real, arr)
    rmse = np.sqrt(mse)
      
    result_parameter= result_parameter.append({'classifiers':'Baysian_AHP','tpr':tpr[1],'fpr':fpr[1],'acc':acc,'MAE':mae,'RMSE':rmse}, ignore_index=True)
       

####################  KNN AHP
    

    knn_fr = ketqua_training[['landslide','KNN_training_AHP','n_KNN_AHP']]
    y_real = ketqua_training[['landslide']]
    
    
    arr = knn_fr[['KNN_training_AHP']] 
    
    print (np.median(np.array(arr)))  ## Trung vi 
    print (np.mean(np.array(arr)))  ## Trung vi 

    pivot =  np.mean(np.array(arr))      
    pivot = 0.910057
    new_arr = np.where(arr > pivot, 1, 0)
 
    ## Tinh FPR, TPR, AUC:     
    fpr, tpr, _ = roc_curve(y_real,  new_arr)
    auc = roc_auc_score(y_real, new_arr)
    acc = metrics.accuracy_score(y_real, new_arr)
    
    
    result_table = result_table.append({'classifiers':'KNN_AHP',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    
    
    y_true, predictions = np.array(y_real), np.array(arr)
    mae = np.mean(np.abs(y_true - predictions))
    mse = mean_squared_error(y_real, arr)
    rmse = np.sqrt(mse)
      
    result_parameter= result_parameter.append({'classifiers':'KNN_AHP','tpr':tpr[1],'fpr':fpr[1],'acc':acc,'MAE':mae,'RMSE':rmse}, ignore_index=True)
       
    # result_parameter.to_csv('c:\\nampam\\3parameter_compare_training.csv', index= False);
    
# Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    
    confusion_matrix = metrics.confusion_matrix(logistic_fr[['landslide']], new_arr)
    fpr, tpr, thresholds = roc_curve(y_real, new_arr)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_real, new_arr, pos_label=2)
        
    auc = metrics.roc_auc_score(y_real, arr)


    fig, ax = plt.subplots(figsize=(6, 6))


    mse = mean_squared_error(Test_y.values.ravel(), predictions)
    
    print ("random forest")
    print("MSE: ", mse)
    print("RMSE: ", np.sqrt(mse))
    rf_MSE.append(mse)
    rf_RMSE.append(np.sqrt(mse))
    
     
def plotHinhVe():
    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)
    
    plt.yticks(np.arange(0.0, 1.1, step=0.1))

    plt.xticks(fontsize=15)

    plt.ylabel("True Positive Rate", fontsize=15)
    
    plt.title('', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    
    plt.show()
     
    #ketqua_testing.to_csv('c:\\nampam\\3ketqua_testing.csv', index= False)

def run_disaster_Data3():
    print ('ok')

##### training    
    ketqua_training = pd.read_csv('c:\\nampam\\3ketqua_trainining_lr_svm.csv' ) 
    cols = list(ketqua_training.columns)
    
    lr_fr = ketqua_training.iloc[:, ketqua_training.columns == 'logistic_training']
    data = lr_fr
    n_lr_fr = (data-np.min(data))/(np.max(data)-np.min(data))

    svm_fr = ketqua_training.iloc[:, ketqua_training.columns == 'svm_training']
    data = svm_fr
    n_svm_fr = (data-np.min(data))/(np.max(data)-np.min(data))

    lr_ahp = ketqua_training.iloc[:, ketqua_training.columns == 'logistic_training_ahp']
    data = lr_ahp
    n_lr_ahp = (data-np.min(data))/(np.max(data)-np.min(data))
    
    svm_ahp = ketqua_training.iloc[:, ketqua_training.columns == 'svm_training_ahp']
    data = svm_ahp
    n_svm_ahp = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_training['n_lr_fr'] =   n_lr_fr
    ketqua_training['n_svm_fr'] =  n_svm_fr
    ketqua_training['n_lr_ahp'] =  n_lr_ahp
    ketqua_training['n_svm_ahp'] = n_svm_ahp

####### testing
    ketqua_testing = pd.read_csv('c:\\nampam\\3ketqua_testing_lr_svm.csv' ) 
    cols = list(ketqua_testing.columns)
    
    lr_fr = ketqua_testing.iloc[:, ketqua_testing.columns == 'logistic_testing']
    data = lr_fr
    n_lr_fr = (data-np.min(data))/(np.max(data)-np.min(data))

    svm_fr = ketqua_testing.iloc[:, ketqua_testing.columns == 'svm_testing']
    data = svm_fr
    n_svm_fr = (data-np.min(data))/(np.max(data)-np.min(data))

    lr_ahp = ketqua_testing.iloc[:, ketqua_testing.columns == 'logistic_testing_ahp']
    data = lr_ahp
    n_lr_ahp = (data-np.min(data))/(np.max(data)-np.min(data))
    
    svm_ahp = ketqua_testing.iloc[:, ketqua_testing.columns == 'svm_testing_ahp']
    data = svm_ahp
    n_svm_ahp = (data-np.min(data))/(np.max(data)-np.min(data))
    
    
    ketqua_testing['n_lr_fr'] =   n_lr_fr
    ketqua_testing['n_svm_fr'] =  n_svm_fr
    ketqua_testing['n_lr_ahp'] =  n_lr_ahp
    ketqua_testing['n_svm_ahp'] = n_svm_ahp


########   all map

    ketqua_allmap = pd.read_csv('c:\\nampam\\3ketqua_allmap_lr_svm.csv' ) 



###########################################


    df = pd.read_csv('c:\\nampam\\3training_part_FR.csv' ) 
    df.columns.values[0] = "class"   
    
    Train_x = df.iloc[:,df.columns !='class']        
    Train_y = df[['class']]
   # df_data = df.iloc[ :,0:15]
    
    testing = pd.read_csv('c:\\nampam\\3testing_part_FR.csv' ) 
    testing.columns.values[0] = "class"   
    Test_x = testing.iloc[:,testing.columns !='class']        
    Test_y = testing[['class']]
    
    all_map= pd.read_csv('c:\\nampam\\np_map_FR.csv' )
    
    
    all_map =  all_map.drop(['OBJECTID','landslide','degree'], axis=1)

    # cols = list(df_data.columns)
    # c1=  cols[-1:]     ## lay phan tu cuoi cung cuar list
    # c2 = cols[:-1]     ##  loai bo phan tu cuoi cung cuar list
    # cols = cols[-1:] + cols[:-1]
    
    # df_data = df_data[cols]
    #df_data.columns.values[0] = "class"   

  
    
    clf = BayesianRidge()
    clf.fit(Train_x, Train_y.values.ravel())

### training
    
    baysian_training_RF = clf.predict(Train_x)            
    data = baysian_training_RF
    n_baysian_rf = (data-np.min(data))/(np.max(data)-np.min(data))
    
    ketqua_training['baysian_training_RF'] =baysian_training_RF
    ketqua_training['n_baysian_rf'] =n_baysian_rf

#testing
    
    baysian_testing_RF = clf.predict(Test_x)            
    data = baysian_testing_RF
    n_baysian_rf = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_testing['baysian_testing_RF'] =baysian_testing_RF
    ketqua_testing['n_baysian_rf'] =n_baysian_rf

### all map

    baysian_all_RF = clf.predict(all_map)            
    ketqua_allmap['baysian_RF'] = baysian_all_RF



################### KNN

    model = neighbors.KNeighborsRegressor(n_neighbors = 2000)
    model.fit(Train_x, Train_y)  #fit the model

##training    
    KNN_training_RF =model.predict(Train_x) #make prediction on test set
    data = KNN_training_RF
    n_KNN_rf = (data-np.min(data))/(np.max(data)-np.min(data))
    
    ketqua_training['KNN_training_RF'] =KNN_training_RF
    ketqua_training['n_KNN_rf'] =n_KNN_rf
    
##3 testing

    KNN_testing_RF =model.predict(Test_x) #make prediction on test set
    data = KNN_testing_RF
    n_KNN_rf = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_testing['KNN_testing_RF'] =KNN_testing_RF
    ketqua_testing['n_KNN_rf'] =n_KNN_rf
    
####  all map

    # KNN_all_RF =model.predict(all_map)
    # ketqua_allmap['KNN_all_RF'] = KNN_all_RF

    df_1 = all_map.iloc[:200000,:]
    KNN_all_RF_1 =model.predict(df_1)
    
    KNN_all_RF = KNN_all_RF_1
    df_2 = all_map.iloc[200000:,:]
    
    for i in range(23):
        df_1 = df_2.iloc[:200000,:]
        df_2 = df_2.iloc[200000:,:]
        
        KNN_all_RF_1 =model.predict(df_1)
        KNN_all_RF = np.concatenate((KNN_all_RF, KNN_all_RF_1))
        
        print (i)
    
    KNN_all_RF_2 =model.predict(df_2)
    
    KNN_all_RF = np.concatenate((KNN_all_RF, KNN_all_RF_2))


    ketqua_allmap['KNN_FR'] = KNN_all_RF    

############### Luu laij ket qua    
    
    ketqua_allmap.to_csv('c:\\nampam\\3ketqua_allmap_lr_svm_bys_knn.csv', index= False)
    ketqua_training.to_csv('c:\\nampam\\3ketqua_training.csv', index= False)
    ketqua_testing.to_csv('c:\\nampam\\3ketqua_testing.csv', index= False)
    
######################


################ chay voi weight la AHP    AHP ########################

    df = 0
    testing =0
    
    df = pd.read_csv('c:\\nampam\\3training_part_AHP.csv' ) 
    df.columns.values[0] = "class"   
    
    
    Train_x = df.iloc[:,df.columns !='class']        
    Train_y = df[['class']]
   # df_data = df.iloc[ :,0:15]
    
    testing = pd.read_csv('c:\\nampam\\3testing_part_AHP.csv' ) 
    testing.columns.values[0] = "class"   
    Test_x = testing.iloc[:,testing.columns !='class']        
    Test_y = testing[['class']]
    
    all_map= pd.read_csv('c:\\nampam\\np_map_AHP.csv' )
    
    
    all_map =  all_map.drop(['OBJECTID','landslide','degree'], axis=1)
    

### baysian

    clf = BayesianRidge()
    clf.fit(Train_x, Train_y.values.ravel())

### training
    
    baysian_training_AHP = clf.predict(Train_x)            
    data = baysian_training_AHP
    n_baysian_AHP = (data-np.min(data))/(np.max(data)-np.min(data))
    
    ketqua_training['baysian_training_AHP'] =baysian_training_AHP
    ketqua_training['n_baysian_AHP'] =n_baysian_AHP

#testing
    
    baysian_testing_AHP = clf.predict(Test_x)            
    data = baysian_testing_AHP
    n_baysian_ANP = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_testing['baysian_testing_AHP'] =baysian_testing_AHP
    ketqua_testing['n_baysian_AHP'] =n_baysian_ANP

### all map

    baysian_all_AHP = clf.predict(all_map)            
    ketqua_allmap['baysian_AHP'] = baysian_all_AHP


################### KNN

    model =0
    model = neighbors.KNeighborsRegressor(n_neighbors = 2000)
    model.fit(Train_x, Train_y)  #fit the model


##training    
    KNN_training_AHP =model.predict(Train_x) #make prediction on test set
    data = KNN_training_AHP
    n_KNN_AHP = (data-np.min(data))/(np.max(data)-np.min(data))
    
    ketqua_training['KNN_training_AHP'] =KNN_training_AHP
    ketqua_training['n_KNN_AHP'] =n_KNN_AHP
    
##3 testing

    KNN_testing_AHP =model.predict(Test_x) #make prediction on test set
    data = KNN_testing_AHP
    n_KNN_AHP = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_testing['KNN_testing_AHP'] =KNN_testing_AHP
    ketqua_testing['n_KNN_AHP'] =n_KNN_AHP
    
####  all map

    # KNN_all_AHP =model.predict(all_map)
    # ketqua_allmap['KNN_all_AHP'] = KNN_all_AHP

    df_1 = all_map.iloc[:200000,:]
    KNN_all_RF_1 =model.predict(df_1)
    
    KNN_all_RF = KNN_all_RF_1
    df_2 = all_map.iloc[200000:,:]
    
    for i in range(23):
        df_1 = df_2.iloc[:200000,:]
        df_2 = df_2.iloc[200000:,:]
        
        KNN_all_RF_1 =model.predict(df_1)
        KNN_all_RF = np.concatenate((KNN_all_RF, KNN_all_RF_1))
        
        print (i)
    
    KNN_all_RF_2 =model.predict(df_2)
    
    KNN_all_RF = np.concatenate((KNN_all_RF, KNN_all_RF_2))


    ketqua_allmap['KNN_AHP'] = KNN_all_RF    


############### Luu laij ket qua    
    
    ketqua_allmap.to_csv('c:\\nampam\\3ketqua_allmap_lr_svm_bys_knn.csv', index= False)
    
    ketqua_training.to_csv('c:\\nampam\\3ketqua_training.csv', index= False)
    ketqua_testing.to_csv('c:\\nampam\\3ketqua_testing.csv', index= False)
    
######################



def run_disaster():
    print ('ok')

##### training    
    ketqua_training = pd.read_csv('c:\\nampam\\ketqua_trainining_lr_svm.csv' ) 
    cols = list(ketqua_training.columns)
    
    lr_fr = ketqua_training.iloc[:, ketqua_training.columns == 'logistic_training']
    data = lr_fr
    n_lr_fr = (data-np.min(data))/(np.max(data)-np.min(data))

    svm_fr = ketqua_training.iloc[:, ketqua_training.columns == 'svm_training']
    data = svm_fr
    n_svm_fr = (data-np.min(data))/(np.max(data)-np.min(data))

    lr_ahp = ketqua_training.iloc[:, ketqua_training.columns == 'logistic_training_ahp']
    data = lr_ahp
    n_lr_ahp = (data-np.min(data))/(np.max(data)-np.min(data))
    
    svm_ahp = ketqua_training.iloc[:, ketqua_training.columns == 'svm_training_ahp']
    data = svm_ahp
    n_svm_ahp = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_training['n_lr_fr'] =   n_lr_fr
    ketqua_training['n_svm_fr'] =  n_svm_fr
    ketqua_training['n_lr_ahp'] =  n_lr_ahp
    ketqua_training['n_svm_ahp'] = n_svm_ahp

####### testing
    ketqua_testing = pd.read_csv('c:\\nampam\\ketqua_testing_lr_svm.csv' ) 
    cols = list(ketqua_testing.columns)
    
    lr_fr = ketqua_testing.iloc[:, ketqua_testing.columns == 'logistic_testing']
    data = lr_fr
    n_lr_fr = (data-np.min(data))/(np.max(data)-np.min(data))

    svm_fr = ketqua_testing.iloc[:, ketqua_testing.columns == 'svm_testing']
    data = svm_fr
    n_svm_fr = (data-np.min(data))/(np.max(data)-np.min(data))

    lr_ahp = ketqua_testing.iloc[:, ketqua_testing.columns == 'logistic_testing_ahp']
    data = lr_ahp
    n_lr_ahp = (data-np.min(data))/(np.max(data)-np.min(data))
    
    svm_ahp = ketqua_testing.iloc[:, ketqua_testing.columns == 'svm_testing_ahp']
    data = svm_ahp
    n_svm_ahp = (data-np.min(data))/(np.max(data)-np.min(data))
    
    
    ketqua_testing['n_lr_fr'] =   n_lr_fr
    ketqua_testing['n_svm_fr'] =  n_svm_fr
    ketqua_testing['n_lr_ahp'] =  n_lr_ahp
    ketqua_testing['n_svm_ahp'] = n_svm_ahp


########   all map

    ketqua_allmap = pd.read_csv('c:\\nampam\\ketqua_allmap_lr_svm.csv' ) 



###########################################


    df = pd.read_csv('c:\\nampam\\training_part_FR.csv' ) 
    df.columns.values[0] = "class"   
    
    Train_x = df.iloc[:,df.columns !='class']        
    Train_y = df[['class']]
   # df_data = df.iloc[ :,0:15]
    
    testing = pd.read_csv('c:\\nampam\\testing_part_FR.csv' ) 
    testing.columns.values[0] = "class"   
    Test_x = testing.iloc[:,testing.columns !='class']        
    Test_y = testing[['class']]
    
    all_map= pd.read_csv('c:\\nampam\\np_map_FR.csv' )
    
    
    all_map =  all_map.drop(['OBJECTID','landslide','degree'], axis=1)

    # cols = list(df_data.columns)
    # c1=  cols[-1:]     ## lay phan tu cuoi cung cuar list
    # c2 = cols[:-1]     ##  loai bo phan tu cuoi cung cuar list
    # cols = cols[-1:] + cols[:-1]
    
    # df_data = df_data[cols]
    #df_data.columns.values[0] = "class"   

  
    
    clf = BayesianRidge()
    clf.fit(Train_x, Train_y.values.ravel())

### training
    
    baysian_training_RF = clf.predict(Train_x)            
    data = baysian_training_RF
    n_baysian_rf = (data-np.min(data))/(np.max(data)-np.min(data))
    
    ketqua_training['baysian_training_RF'] =baysian_training_RF
    ketqua_training['n_baysian_rf'] =n_baysian_rf

#testing
    
    baysian_testing_RF = clf.predict(Test_x)            
    data = baysian_testing_RF
    n_baysian_rf = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_testing['baysian_testing_RF'] =baysian_testing_RF
    ketqua_testing['n_baysian_rf'] =n_baysian_rf

### all map

    baysian_all_RF = clf.predict(all_map)            
    ketqua_allmap['baysian_RF'] = baysian_all_RF



################### KNN

    model = neighbors.KNeighborsRegressor(n_neighbors = 2000)
    model.fit(Train_x, Train_y)  #fit the model

##training    
    KNN_training_RF =model.predict(Train_x) #make prediction on test set
    data = KNN_training_RF
    n_KNN_rf = (data-np.min(data))/(np.max(data)-np.min(data))
    
    ketqua_training['KNN_training_RF'] =KNN_training_RF
    ketqua_training['n_KNN_rf'] =n_KNN_rf
    
##3 testing

    KNN_testing_RF =model.predict(Test_x) #make prediction on test set
    data = KNN_testing_RF
    n_KNN_rf = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_testing['KNN_testing_RF'] =KNN_testing_RF
    ketqua_testing['n_KNN_rf'] =n_KNN_rf
    
####  all map


    df_1 = all_map.iloc[:200000,:]
    KNN_all_RF_1 =model.predict(df_1)
    
    KNN_all_RF = KNN_all_RF_1
    df_2 = all_map.iloc[200000:,:]
    
    for i in range(23):
        df_1 = df_2.iloc[:200000,:]
        df_2 = df_2.iloc[200000:,:]
        
        KNN_all_RF_1 =model.predict(df_1)
        KNN_all_RF = np.concatenate((KNN_all_RF, KNN_all_RF_1))
        
        print (i)
    
    KNN_all_RF_2 =model.predict(df_2)
    
    KNN_all_RF = np.concatenate((KNN_all_RF, KNN_all_RF_2))


    ketqua_allmap['KNN_RF'] = KNN_all_RF


############### Luu laij ket qua    
    
    ketqua_allmap.to_csv('c:\\nampam\\ketqua_allmap_lr_svm_bys_knn.csv', index= False)
    ketqua_training.to_csv('c:\\nampam\\ketqua_training.csv', index= False)
    ketqua_testing.to_csv('c:\\nampam\\ketqua_testing.csv', index= False)
    
######################


    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    
    ## point all
    predictions = clf.predict(Test_x)        
    
    
################ chay voi weight la AHP    AHP ########################

    df = 0
    testing =0
    
    df = pd.read_csv('c:\\nampam\\training_part_AHP.csv' ) 
    df.columns.values[0] = "class"   
    
    
    Train_x = df.iloc[:,df.columns !='class']        
    Train_y = df[['class']]
   # df_data = df.iloc[ :,0:15]
    
    testing = pd.read_csv('c:\\nampam\\testing_part_AHP.csv' ) 
    testing.columns.values[0] = "class"   
    Test_x = testing.iloc[:,testing.columns !='class']        
    Test_y = testing[['class']]
    
    all_map= pd.read_csv('c:\\nampam\\np_map_AHP.csv' )
    
    
    all_map =  all_map.drop(['OBJECTID','landslide','degree'], axis=1)
    

### baysian

    clf = BayesianRidge()
    clf.fit(Train_x, Train_y.values.ravel())

### training
    
    baysian_training_AHP = clf.predict(Train_x)            
    data = baysian_training_AHP
    n_baysian_AHP = (data-np.min(data))/(np.max(data)-np.min(data))
    
    ketqua_training['baysian_training_AHP'] =baysian_training_AHP
    ketqua_training['n_baysian_AHP'] =n_baysian_AHP

#testing
    
    baysian_testing_AHP = clf.predict(Test_x)            
    data = baysian_testing_AHP
    n_baysian_ANP = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_testing['baysian_testing_AHP'] =baysian_testing_AHP
    ketqua_testing['n_baysian_ANP'] =n_baysian_ANP

### all map

    baysian_all_AHP = clf.predict(all_map)            
    ketqua_allmap['baysian_AHP'] = baysian_all_AHP


################### KNN

    model =0
    model = neighbors.KNeighborsRegressor(n_neighbors = 2000)
    model.fit(Train_x, Train_y)  #fit the model


##training    
    KNN_training_AHP =model.predict(Train_x) #make prediction on test set
    data = KNN_training_AHP
    n_KNN_AHP = (data-np.min(data))/(np.max(data)-np.min(data))
    
    ketqua_training['KNN_training_AHP'] =KNN_training_AHP
    ketqua_training['n_KNN_AHP'] =n_KNN_AHP
    
##3 testing

    KNN_testing_AHP =model.predict(Test_x) #make prediction on test set
    data = KNN_testing_AHP
    n_KNN_AHP = (data-np.min(data))/(np.max(data)-np.min(data))

    ketqua_testing['KNN_testing_AHP'] =KNN_testing_AHP
    ketqua_testing['n_KNN_AHP'] =n_KNN_AHP
    
####  all map

    # KNN_all_AHP =model.predict(all_map)
    # ketqua_allmap['KNN_AHP'] = KNN_all_AHP


    df_1 = all_map.iloc[:200000,:]
    KNN_all_RF_1 =model.predict(df_1)
    
    KNN_all_RF = KNN_all_RF_1
    df_2 = all_map.iloc[200000:,:]
    
    for i in range(23):
        df_1 = df_2.iloc[:200000,:]
        df_2 = df_2.iloc[200000:,:]
        
        KNN_all_RF_1 =model.predict(df_1)
        KNN_all_RF = np.concatenate((KNN_all_RF, KNN_all_RF_1))
        
        print (i)
    
    KNN_all_RF_2 =model.predict(df_2)
    
    KNN_all_RF = np.concatenate((KNN_all_RF, KNN_all_RF_2))


    ketqua_allmap['KNN_AHP'] = KNN_all_RF


############### Luu laij ket qua    
    
    ketqua_allmap.to_csv('c:\\nampam\\ketqua_allmap_lr_svm_bys_knn.csv', index= False)
    
    ketqua_training.to_csv('c:\\nampam\\ketqua_training.csv', index= False)
    ketqua_testing.to_csv('c:\\nampam\\ketqua_testing.csv', index= False)
    
######################













    
    ##3 DT

    dt_model = DecisionTreeRegressor(max_depth=100,max_leaf_nodes=100)
    dt_model.fit(Train_x, Train_y)
    predict_dt = dt_model.predict(Train_x)
def run_compare(df):
    dbfrr = Dbf5('c:\\nampam\\Export_Output.dbf')
    df = dbfrr.to_dataframe()
    table =0
    
    table = DBF('c:\\nampam\\Export_Output.dbf',load = True)
    datatable = table.to_csv
    
    r,c = df.shape
    X = df.iloc[:,df.columns !='class']
    features = X.columns.tolist()
    y = df[['class']]

   # _random_state =  random.randint(0,100000)

    
    
  #  print (_random_state)
    dt_MSE =list()
    dt_RMSE= list()
    logistic_MSE =list()
    logistic_RMSE= list()
    linear_MSE =list()
    linear_RMSE= list()
    baysian_MSE =list()
    baysian_RMSE= list()
    svm_MSE =list()
    svm_RMSE= list()
    rf_MSE =list()
    rf_RMSE= list()
   
    
    
    for i in range (10):
        
        Train_x, Test_x, Train_y, Test_y = train_test_split(X, y, train_size=0.7,random_state = 1)

        # model_reg= DecisionTreeRegressor()        
        # model_reg.fit(Train_x, Train_y.values.ravel())            
        # predictions = model_reg.predict(Test_x)
        # score = model_reg.score(Train_x, Train_y.values.ravel())
        # #print("R-squared:", score)        
        # mse = mean_squared_error(Test_y.values.ravel(), predictions)
        # print("decision tree")
        # print("MSE: ", mse)
        # print("RMSE: ", np.sqrt(mse))
        
        # dt_MSE.append(mse)
        # dt_RMSE.append(np.sqrt(mse))

        # # logistic
        # pipe = make_pipeline(StandardScaler(), LogisticRegression())    
        # pipe.fit(Train_x, Train_y.values.ravel())
        # predictions = pipe.predict(Test_x)        
        # mse = mean_squared_error(Test_y.values.ravel(), predictions)    
        
        # # model_logistic = LogisticRegression(solver= 'lbfgs', max_iter=4000)        
        # # model_logistic.fit(Train_x, Train_y)
        # # predictions = model_logistic.predict(Test_x)        
        # # mse = mean_squared_error(Test_y.values.ravel(), predictions)        
    
        # print("logistic")
        # print("MSE: ", mse)
        # print("RMSE: ", np.sqrt(mse))        
        # logistic_MSE.append(mse)
        # logistic_RMSE.append(np.sqrt(mse))
    
        
#         X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
#         # weight (kg)
#         y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
        
# #       (15 ;132); (20 ; 150); (35;160); (39;162); (51;149); (65;170)

#         Train_x =np.array([[15, 20, 35, 39, 51,6 5]]).T             
#         Train_y = np.array([[132,150, 160,162,149,170]]).T
        
        # linear regression        
        
        # doc du lieu tu all Nampam
        
        df_all = pd.read_csv('c:\\nampam\\np_map.csv' ) 
        
        Test_x = df_all.iloc[ : , 2:15]
        ketqua_all = df_all.iloc[:, 0:2]

        Train_x = df_data.iloc[:,df_data.columns !='class']        
        Train_y = df_data[['class']]

        
    
        model_linear  = LinearRegression()                
        model_linear.fit(Train_x, Train_y)
        score = model_linear.score(Train_x, Train_y.values.ravel())        
        #print("R-squared:", score)
        
        ## dua ra ket qua cua Training 
        predictions = model_linear.predict(Train_x)                
        data_1 = df_data
        data_1["linear"] =predictions
        
        
        mse = mean_squared_error(Test_y.values.ravel(), predictions)
        
        
        ## dua ra ket qua cua all point
        predictions = model_linear.predict(Test_x)                
        mse = mean_squared_error(Test_y.values.ravel(), predictions)
        ketqua_all["linear"] = predictions
        
        ketqua_all.to_csv('c:\\nampam\\ketqua.csv', index= False);
        
        print("linear")
        print("MSE: ", mse)
        print("RMSE: ", np.sqrt(mse))
        print(model_linear.coef_)              # cac he so beta 
        print(model_linear.intercept_)         # he so chan tren 
        linear_MSE.append(mse)
        linear_RMSE.append(np.sqrt(mse))

        
        predictions =0.1
        model_logistic = LogisticRegression(solver='lbfgs' , max_iter=100, multi_class = 'multinomial')        
        #model_logistic = LogisticRegression()        
        model_logistic.fit(Train_x, Train_y)
        
        predictions = model_logistic.predict_proba(Test_x)
        
        prediction = predictions[:,1]
        ketqua_all["lr1"] =prediction
        ketqua_all["lr2"] =predictions[:,0]
                
        predictions = model_logistic.predict(Test_x)

        mse = mean_squared_error(test_y.values.ravel(), predictions)        
        
        #baysian ridge
        clf = BayesianRidge()
        clf.fit(Train_x, Train_y.values.ravel())
        
        predictions = clf.predict(Train_x)        
        
        ## point all
        predictions = clf.predict(Test_x)        
        
        
        ketqua_all["baysian"] = predictions        
        
        
        mse = mean_squared_error(Test_y.values.ravel(), predictions)
        
        print("baysian ridge")
        print("MSE: ", mse)
        print("RMSE: ", np.sqrt(mse))
        baysian_MSE.append(mse)
        baysian_RMSE.append(np.sqrt(mse))

      
        
        
        ## SVM 
        # Fit regression model
        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        #svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,  coef0=1)
        svr_rbf.fit(Train_x, Train_y.values.ravel())
        
        svr_lin = SVR(kernel='linear')
        svr_lin.fit(Train_x, Train_y.values.ravel())
        
        ## point all
        predictions = svr_rbf.predict(Test_x)        
        ketqua_all["SVM"] = predictions        
        
        
        
        mse = mean_squared_error(Test_y.values.ravel(), predictions)        
        print("SVM")
        print("MSE: ", mse)
        print("RMSE: ", np.sqrt(mse))        
        svm_MSE.append(mse)
        svm_RMSE.append(np.sqrt(mse))
        
        
        
        #### decision tree regression
        dt_model = DecisionTreeRegressor(max_depth=100,max_leaf_nodes=100)
        dt_model.fit(Train_x, Train_y)
        predict_dt = dt_model.predict(Test_x)
        
        
        
        
        
        #random forest        
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(Train_x, Train_y.values.ravel())
        predictions = regressor.predict(Test_x)
        ketqua_all["RF"] = predictions        

        
        mse = mean_squared_error(Test_y.values.ravel(), predictions)
        print ("random forest")
        print("MSE: ", mse)
        print("RMSE: ", np.sqrt(mse))
        rf_MSE.append(mse)
        rf_RMSE.append(np.sqrt(mse))
      
        
        base_cls = DecisionTreeRegressor()             
                
        regr = BaggingRegressor(base_estimator = base_cls, n_estimators=10, random_state=0).fit(Train_x, Train_y)
    
        predictions = regr.predict(Test_x)
        
    print ("Decision tree: MSE va RMSE: ", np.mean(dt_MSE), ' : ',np.mean(dt_RMSE))
    print ("Logistic: MSE va RMSE: ", np.mean(logistic_MSE), ' : ',np.mean(logistic_RMSE))
    print ("Linear: MSE va RMSE: ", np.mean(linear_MSE), ' : ',np.mean(linear_RMSE))
    print ("Baysian: MSE va RMSE: ", np.mean(baysian_MSE), ' : ',np.mean(baysian_RMSE))
    print ("SVM: MSE va RMSE: ", np.mean(svm_MSE), ' : ',np.mean(svm_RMSE))
    print ("Random forest: MSE va RMSE: ", np.mean(rf_MSE), ' : ',np.mean(rf_RMSE))

    names =('DT','Log','Linear','Baysian','SVM','RF') 

    results =[]
    results.append(dt_MSE)
    results.append(logistic_MSE)
    results.append(linear_MSE)
    results.append(baysian_MSE)    
    results.append(svm_MSE)
    results.append(rf_MSE)
    
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
  #  ax = fig.add_subplot(111)
#    plt.boxplot(results)
    plt.boxplot(results, labels=names, showmeans=True)

    #ax.set_xticklabels(names)
    plt.show()    
    
    return 
    


def Chuyendoi():
    
    cols = list(df.columns)
    c1=  cols[-1:]     ## lay phan tu cuoi cung cuar list
    c2 = cols[:-1]     ##  loai bo phan tu cuoi cung cuar list
    cols = cols[-1:] + cols[:-1]
    
    df = df[cols]
    df.columns.values[0] = "class"   

#    df.to_csv('c:\\baitap_datamining\\wine.csv', index= False);
#    print (df)
    # Categorical boolean mask
    categorical_feature_mask = df.dtypes==object
   # filter categorical columns using mask and turn it into a list
    categorical_cols = df.columns[categorical_feature_mask].tolist()
    
    le = LabelEncoder()
    
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    df[categorical_cols].head(10)    

        

    
    
def main():
  #  gin = gini_split(5, 9, gini(2, 3), gini(7, 2))
    #Golf()   
    df = pd.read_csv('c:\\nampam\\training_part.csv' ) 
    
    df_data = df.iloc[ :,0:15]
    
      
    # cols = list(df_data.columns)
    # c1=  cols[-1:]     ## lay phan tu cuoi cung cuar list
    # c2 = cols[:-1]     ##  loai bo phan tu cuoi cung cuar list
    # cols = cols[-1:] + cols[:-1]
    
    # df_data = df_data[cols]
    df_data.columns.values[0] = "class"   


    # df = pd.read_csv('c:\\baitap_datamining\\data50K_regression.csv', header = None ) 

    # df.rename(columns = {0:'class'}, inplace = True)

    #df.columns.values[0] = "class"   
    
    run_compare(df)
    
    return

if __name__ == "__main__":
    main()
        