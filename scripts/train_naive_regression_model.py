# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:16:43 2016

@author: rakhunzy
"""


import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl

import scipy
from scipy import signal
from scipy import stats

# In[]

def norm_cdf(mean=0.0, std=1.0):
    x = np.linspace(0, 599, 600)
    y = stats.norm.cdf(x, loc=mean, scale=std)
    return x, y

def gen_cdf(volume,k):
    mean, std = volume, k
    return norm_cdf(mean, std)

def CRPS(label, pred):
    """ Custom evaluation metric on CRPS.
    """
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    
    return np.sum(np.square(label - pred)) / label.size


def calc_CRPS(pred_volumes, truth_volumes):
    _, cdf_pred = gen_cdf(np.array(pred_volumes), 0.21)
    _, cdf_data = gen_cdf(np.array(truth_volumes),0.0001)      
    return CRPS(np.array(cdf_data),np.array(cdf_pred))

def calc_CRPS_best(pred_volumes, truth_volumes):  
    CRPS_n = []
    step = 0.2
    n_min = 0.1
    n_max = 50.1
    count = int((n_max - n_min) / step)
    for n in xrange(1, count, 1):
        label = []
        pred = []
        for i in range(len(pred_volumes)):
            #k_dias = (1.0 - ((100.0 - pred_volumes[i]) / 100.0))**0.95
            _, cdf_diastola = gen_cdf(pred_volumes[i],((n_min + n*step)))
            pred.append(cdf_diastola)
            _, cdf_data_diastola = gen_cdf(float(truth_volumes[i]),0.01)
            label.append(cdf_data_diastola)
        CRPS_n.append([CRPS(np.array(label),np.array(pred)), n_min + n*step])
        print CRPS_n[-1]
    CRPS_n = np.array(CRPS_n)
    argmin = np.argmin(CRPS_n, axis=0)
    print CRPS_n[argmin[0]]
    return CRPS_n[argmin[0]][0]

# In[]​

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def read_dataset(filename):
    train_df = pd.read_csv('train.csv')
    
    sys_min, sys_mean, sys_max = np.percentile(train_df['Systole'],[1,50,99])
    dias_min, dias_mean, dias_max = np.percentile(train_df['Diastole'],[1,50,99])
    
    ids = np.array([])
    X = np.array([])
    Y = np.array([])    

    with open(filename, 'rb') as csvfile:
        fieldnames = ['Patient_Name',
                  'Is_Man', 'Is_Adult',
                  'Age_Y', 'Is_Aged_Y', 'Is_Aged_M', 'Is_Aged_W', 'Is_Aged_D',
                  'V_min', 'V_max',
                  'V_min_CPR', 'V_max_CRP',
                  'H_lv',
                  'Sax_Count', 'Sax_Waste_Count',
                  'Time_sys_dias', 'Time_sys_dias_CPR',
                  'Min_Area_sys', 'Avr_Area_sys', 'Max_Area_sys',
                  'Min_Area_dias', 'Avr_Area_dias', 'Max_Area_dias',
                  'Min_Area_sys_CPR', 'Avr_Area_sys_CPR', 'Max_Area_sys_CPR',
                  'Min_Area_dias_CPR', 'Avr_Area_dias_CPR', 'Max_Area_dias_CPR',
                ]
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, lineterminator='\n')
    
        for row in reader:
            sample = np.array([float(row[fn]) for fn in fieldnames])

#            sample = np.append(sample, [(sample[10]*sample[8])**0.5, (sample[9]*sample[11])**0.5])
#            sample = np.append(sample, [min(sample[10],sample[8]), max(sample[9],sample[11])])
#            sample = np.append(sample, [max(sample[10],sample[8]), max(sample[9],sample[11])])
            
            sample[fieldnames.index('V_min')] = clamp(sample[fieldnames.index('V_min')],sys_min,sys_max)
            sample[fieldnames.index('V_min_CPR')] = clamp(sample[fieldnames.index('V_min_CPR')],sys_min,sys_max)
            
            sample[fieldnames.index('V_max')] = clamp(sample[fieldnames.index('V_max')],dias_min,dias_max)
            sample[fieldnames.index('V_max_CRP')] = clamp(sample[fieldnames.index('V_max_CRP')],dias_min,dias_max)     
            
            sample = np.append(sample, [(sample[10]+sample[8])/2, (sample[9]+sample[11])/2])
            
#            print sample
            sample_sys  = np.append(sample, 0)
            sample_dias = np.append(sample, 1)
            
            v_sys = float(train_df[train_df['Id']==sample[0]]['Systole'])
            v_dias = float(train_df[train_df['Id']==sample[0]]['Diastole'])   
            
            if len(X):               
                X = np.vstack((X, sample_sys))
                X = np.vstack((X, sample_dias))
                Y = np.vstack((Y, np.array([[v_sys],[v_dias]])))
                ids = np.vstack((ids, np.array([[sample[0]],[sample[0]]])))
            else:
                X = np.array([sample_sys])               
                X = np.vstack((X, sample_dias))
                Y = np.array([[v_sys],[v_dias]])
                ids = np.array([[sample[0]],[sample[0]]])
                
#            print patient_name, sample
    fieldnames.append('mean_sys')
    fieldnames.append('mean_dias')
#    fieldnames.append('hmean_sys')
#    fieldnames.append('hmean_dias')
#    fieldnames.append('min_sys')
#    fieldnames.append('min_dias')
#    fieldnames.append('max_sys')
#    fieldnames.append('max_dias')
    fieldnames.append('is_diastole')
    column_names = fieldnames
#    hist_len = len(X[0])-len(column_names)
#    for i in range(hist_len):
#        column_names.append('h'+str(i))
    return X, Y, ids, column_names

#if __name__ == "__main__":

# In[]
plot_cool_scatter_matrix = False
if plot_cool_scatter_matrix:
    sns.set(style="white")
    
    dfX = pd.DataFrame(X, index=indices, columns=column_names[1:])
    dfY = pd.DataFrame(Y, index=indices, columns=['Y'])
    df = pd.concat([dfX, dfY], axis=1)
    
    g = sns.PairGrid(df[['Age_Y', 'mean_sys', 'mean_dias','Y']], diag_sharey=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_upper(plt.scatter)
    g.map_diag(sns.kdeplot, lw=3)
    
# In[]  
X, Y, ids, column_names = read_dataset('data_for_model_500_v2.csv')
#selected_cols=[1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29]
indices = X[:,0]
X = X[:,1:]

X_df = pd.read_csv('data_for_model_500_v2.csv', names=column_names, skiprows=0)

# In[]
from sklearn import ensemble 
X_train, X_test, y_train, y_test, idx_train, idx_test = skl.cross_validation.train_test_split(X, Y.ravel(), indices, test_size=0.2, random_state=0)

clf = skl.ensemble.GradientBoostingRegressor(n_estimators = 1000, learning_rate=0.1, max_depth=4, random_state=0, loss='ls', subsample=1.0)

clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)

# In[]
crps = calc_CRPS_best(y_pred.ravel(), y_test.ravel())
print(crps)

idx_test[np.abs(y_pred-y_test)>50]

# In[]
plt.plot((0,400),(0,400))
plt.scatter(np.array(y_pred)[X_test[:,-1]==1], np.array(y_test)[X_test[:,-1]==1], color='g')
plt.scatter(np.array(y_pred)[X_test[:,-1]==0], np.array(y_test)[X_test[:,-1]==0], color='b')
plt.grid(True)
plt.ylim((0,400))
plt.xlim((0,400))
plt.xticks(range(0,400,25))
plt.yticks(range(0,400,25))
#plt.plot(range(len(y_pred)), y_test)
#plt.plot(range(len(y_pred)), y_pred)
#plt.plot(range(len(y_pred)), y_test - y_pred)

# In[]
print clf.feature_importances_, column_names

# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
#plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(np.array(column_names)[1:])[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()