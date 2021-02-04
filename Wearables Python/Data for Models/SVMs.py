#Packages for data analysis
import numpy as np
import pandas as pd

import scipy.io
from sklearn import svm

#Visual your data
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# SVM
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

# Get data
data_complete = scipy.io.loadmat('FeatureMatrices.mat')
labels = data_complete['FMinfo']
test = data_complete['WFM_1']
clas = test[:, 129]

x = np.delete(test, 129, axis=1)
y = clas
# print(x)

# Variables
num_features = 10
dict_features = {}
times = 10

# Create dictionary
for i in range (1, num_features+1):
    dict_features[i] = 0

# Calculating the best model 10 times
print('Best features')
for j in range (1, times+1):
    best_accuracy = 0
    best_kernel = ''
    best_features = 0
    for i in range(1, num_features+1):
        # Las k mejores características
        X_new = SelectKBest(chi2, k=num_features).fit_transform(abs(x), y)

        # Split the data for train and test
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3) 
        #print(X_train)

        # Standardization of data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)

        # RBF, Polynomial and Linear Kernel
        # rbf = svm.SVC(kernel='rbf', gamma=0.6, C=1).fit(X_train, y_train)
        # poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
        # linear = svm.SVC(kernel='linear').fit(X_train, y_train)

        # RBF, Polynomial and Linear Kernel with standardization
        rbf = svm.SVC(kernel='rbf', gamma=0.6, C=1).fit(X_scaled, y_train)
        poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_scaled, y_train)
        linear = svm.SVC(kernel='linear').fit(X_scaled, y_train)

    #     poly_pred_test = poly.predict(X_test)
    #     rbf_pred_test = rbf.predict(X_test)
    #     linear_pred_test = linear.predict(X_test)

        poly_pred_test = cross_val_predict(poly, X_test, y_test, cv = 5)
        rbf_pred_test = cross_val_predict(rbf, X_test, y_test, cv = 5)
        linear_pred_test = cross_val_predict(linear, X_test, y_test, cv = 5)
        #print(linear_pred_test)

        # Accuracy test
        poly_accuracy = accuracy_score(y_test, poly_pred_test)
        poly_f1 = f1_score(y_test, poly_pred_test, average='weighted')

        rbf_accuracy = accuracy_score(y_test, rbf_pred_test)
        rbf_f1 = f1_score(y_test, rbf_pred_test, average='weighted')

        linear_accuracy = accuracy_score(y_test, linear_pred_test)
        linear_f1 = f1_score(y_test, linear_pred_test, average='weighted')

        if poly_accuracy > rbf_accuracy and poly_accuracy > linear_accuracy and poly_accuracy > best_accuracy:
            best_accuracy = poly_accuracy
            best_kernel = 'Poly'
            best_features = i
        elif rbf_accuracy > poly_accuracy and rbf_accuracy > linear_accuracy and rbf_accuracy > best_accuracy:
            best_accuracy = rbf_accuracy
            best_kernel = 'RBF'
            best_features = i
        elif linear_accuracy > poly_accuracy and linear_accuracy > rbf_accuracy and linear_accuracy > best_accuracy:
            best_accuracy = linear_accuracy
            best_kernel = 'Linear'
            best_features = i
    dict_features[best_features] = dict_features[best_features] + 1
    print(dict_features)
        
# print('Best Accuracy: ', "%.2f" % (best_accuracy*100))
# print('Best Kernel: ', best_kernel)
# print('Number of features: ', best_features)
print('----- Final Result -----')
print(dict_features)
