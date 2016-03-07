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
import xgboost 
import xgboost as xgb

from sklearn import ensemble 

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


def calc_CRPS(pred_volumes, truth_volumes, sigma):
    _, cdf_pred = gen_cdf(np.array(pred_volumes), sigma)
    _, cdf_data = gen_cdf(np.array(truth_volumes),0.0001)      
    return CRPS(np.array(cdf_data),np.array(cdf_pred))

def calc_CRPS_best(pred_volumes, truth_volumes):  
    CRPS_n = []
    step = 0.5
    n_min = 0.1
    n_max = 30.1
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

        if (len(CRPS_n) > 3):
            if (np.all(np.argsort(np.array(CRPS_n)[-3:,0]) == np.array([0,1,2]))):
                break
        
#        print CRPS_n[-1]
        
    CRPS_n = np.array(CRPS_n)
    argmin = np.argmin(CRPS_n, axis=0)
#    print CRPS_n[argmin[0]]
    return CRPS_n[argmin[0]]

def calc_crps(y_preds, y_tests):
    crps = []
    for y_pred, y_test in zip(y_preds, y_tests):
        crps_, sigma = calc_CRPS_best(y_pred.ravel(), y_test.ravel())
        crps.append(crps_)
    print(crps)
    return crps

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

    X = X[:,1:]
    column_names = column_names[1:]
    
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
X, Y, indices, column_names = read_dataset('data_for_model_train_v2.csv')
#selected_cols=[1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29]
#indices = X[:,0]

X_df = pd.read_csv('data_for_model_train_v2.csv', names=column_names, skiprows=0)

# In[]

def train_regressors(X, Y, folds, regressor_factory):
    regressors = []
    y_preds = []
    y_tests = []
    test_ids = []
    
    i = 1
    for train_id, test_id in skf:
        print("Training fold %s" % (i))
        X_train = X[train_id]
        X_test = X[test_id]
        y_train = Y[train_id]
        y_test = Y[test_id]
        
        reg = regressor_factory()
        regressors.append(reg)
        reg.fit(X_train, y_train.ravel()) 
        y_pred = reg.predict(X_test)
        y_preds.append(y_pred.ravel())
        y_tests.append(y_test.ravel())
        test_ids.append(test_id.ravel())
        
        i = i + 1
    return regressors, y_preds, y_tests, test_ids

def xgb_factory():
    return xgb.XGBRegressor(max_depth = 3, learning_rate = 0.1, n_estimators = 1000, seed = 0, objective="reg:linear")

def gbt_factory():
    return skl.ensemble.GradientBoostingRegressor(n_estimators = 1000, learning_rate=0.1, max_depth=4, random_state=0, loss='huber', subsample=1.0)

def adb_factory():
    return skl.ensemble.AdaBoostRegressor(base_estimator = skl.tree.DecisionTreeRegressor(max_depth=4), n_estimators=1000, learning_rate = 1., random_state=0, loss='linear')


# In[]
skf = skl.cross_validation.LabelKFold(map(int, indices), n_folds = 10)

regressors_xgb, y_preds_xgb, y_tests_xgb, test_ids_gxb = train_regressors(X, Y, skf, xgb_factory)
regressors_gbt, y_preds_gbt, y_tests_gbt, test_ids_gbt = train_regressors(X, Y, skf, gbt_factory)
regressors_adb, y_preds_adb, y_tests_adb, test_ids_adb = train_regressors(X, Y, skf, adb_factory)

# In[]
crps_xgb = calc_crps(y_preds_xgb, y_tests_xgb)
crps_gbt = calc_crps(y_preds_gbt, y_tests_gbt)
crps_adb = calc_crps(y_preds_adb, y_tests_adb)

# In[]
t = []
for a,b in zip(y_preds_xgb, y_preds_gbt):
    t.append((a.ravel()+b.ravel())/2)
crps_xgbgbt = calc_crps(t, y_tests_gbt)

y_preds_xgb

# In[]

#idx_test[np.abs(y_pred-y_test)>50]

y_preds = t
y_tests = y_tests_xgb

y_pred =  (np.concatenate(y_preds_xgb).ravel() + np.concatenate(y_preds_gbt).ravel()) / 2 #np.concatenate(y_preds).ravel()
y_test = np.concatenate(y_tests).ravel()
test_id = np.concatenate(test_ids).ravel()

# In[]
prediction_errors = np.abs(y_pred - y_test)
bad_patients = np.transpose(np.array([indices[test_id[prediction_errors > 50]], prediction_errors[prediction_errors > 50]]))
bad_patients = bad_patients[np.argsort(bad_patients[:,0]),:]

# In[]
plt.figure()
plt.plot((0,400),(0,400))
plt.axes().set_aspect('equal', 'datalim')
plt.scatter(y_test[X[test_id,-1] == 1], y_pred[X[test_id,-1] == 1], color='g')
plt.scatter(y_test[X[test_id,-1] == 0], y_pred[X[test_id,-1] == 0], color='b')
plt.grid(True)
plt.xlabel('truth')
plt.ylabel('pred')
plt.ylim((0,400))
plt.xlim((0,400))
plt.xticks(range(0,400,25))
plt.yticks(range(0,400,25))

calc_CRPS_best(y_pred.ravel(), y_test.ravel())

# In[] Generate submission
​
X_validate, Y_validate, ids_validate, column_names = read_dataset('data_for_model_validate_v2.csv')

X_validate = filter_X(X_validate, Y_validate, sis_dis_min, sis_dis_mean, sis_dis_max)
Y_pred = clf.predict(X_validate)
​
# In[]
#plt.plot(range(len(Y_pred)/2), Y_pred[X_validate[:,0] == 1].ravel(), color = '#0000bb')
#plt.plot(range(len(Y_pred)/2), Y_pred[X_validate[:,0] == 0].ravel(), color = '#00bb00')
#plt.plot(range(len(Y_pred)/2), Y_pred_80[X_validate[:,0] == 1].ravel(), color = '#000044')
#plt.plot(range(len(Y_pred)/2), Y_pred_80[X_validate[:,0] == 0].ravel(), color = '#004400')
​
​
# In[]
​
#plt.plot(range(len(Y_pred)/2), Y_pred_80.ravel()- Y_pred.ravel(),'r')
plt.grid(True)
write_submission_file("submission_naive_features_full_ds.csv", X_validate, Y_pred.ravel(), ids_validate.ravel(), optimal_sigma)
​
Y_pred[X_validate[:,0] == 1]
Y_pred[X_validate[:,0] == 0]
​
# In[]
​
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
#plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(column_names)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


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