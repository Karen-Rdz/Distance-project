#!/usr/bin/env python
# coding: utf-8

# In[1]:


#packages for data analysis
import numpy as np
import pandas as pd

import scipy.io
from sklearn import svm

#visual your data
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# get data
data_complete = scipy.io.loadmat('FeatureMatrices.mat')
labels = data_complete['FMinfo']
test = data_complete['WFM_1']
clas = test[:, 129]

x = np.delete(test, 129, axis=1)
y = clas

print(x)


# In[29]:


# Variables
num_features = 10
best_accuracy = 0
best_kernel = ''
best_features = 0
dict_features = {}
times = 10

# Create dictionary
for i in range (1, num_features+1):
    dict_features[i] = 0

# Calculating the best model 10 times
for j in range (1, times+1):
    for i in range(1, num_features+1):
        # Las k mejores características
        X_new = SelectKBest(chi2, k=num_features).fit_transform(abs(x), y)

        # Split the data for train and test
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3) 
        #print(X_train)

        # RBF, Polynomial and Linear Kernel
        rbf = svm.SVC(kernel='rbf', gamma=0.6, C=1).fit(X_train, y_train)
        poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
        linear = svm.SVC(kernel='linear').fit(X_train, y_train)

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
        
print('Best Accuracy: ', "%.2f" % (best_accuracy*100))
print('Best Kernel: ', best_kernel)
print('Number of features: ', best_features)
print(dict_features)


# In[4]:


# Las k mejores características
X_new = SelectKBest(chi2, k=8).fit_transform(abs(x), y)

# Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3) 
print(X_train)


# In[5]:


# RBF, Polynomial and Linear Kernel
rbf = svm.SVC(kernel='rbf', gamma=0.6, C=1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
linear = svm.SVC(kernel='linear').fit(X_train, y_train)


# In[6]:


# Predict values
# poly_pred_test = poly.predict(X_test)
# rbf_pred_test = rbf.predict(X_test)
# linear_pred_test = linear.predict(X_test)

# poly_pred_train = poly.predict(X_train)
# rbf_pred_train = rbf.predict(X_train)
# linear_pred_train = linear.predict(X_train)

poly_pred_test = cross_val_predict(poly, X_test, y_test, cv = 5)
rbf_pred_test = cross_val_predict(rbf, X_test, y_test, cv = 5)
linear_pred_test = cross_val_predict(linear, X_test, y_test, cv = 5)

poly_pred_train = cross_val_predict(poly, X_train, y_train, cv = 5)
rbf_pred_train = cross_val_predict(rbf, X_train, y_train, cv = 5)
linear_pred_train = cross_val_predict(linear, X_train, y_train, cv = 5)

print(linear_pred_test)


# In[7]:


# Accuracy test
print("-----Accuracy Test-----")
poly_accuracy = accuracy_score(y_test, poly_pred_test)
poly_f1 = f1_score(y_test, poly_pred_test, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100), '\n')

rbf_accuracy = accuracy_score(y_test, rbf_pred_test)
rbf_f1 = f1_score(y_test, rbf_pred_test, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100), '\n')

linear_accuracy = accuracy_score(y_test, linear_pred_test)
linear_f1 = f1_score(y_test, linear_pred_test, average='weighted')
print('Accuracy (Linear Kernel): ', "%.2f" % (linear_accuracy*100))
print('F1 (Linear Kernel): ', "%.2f" % (linear_f1*100), '\n')

# Accuracy train
print("-----Accuracy Train-----")
poly_accuracy = accuracy_score(y_train, poly_pred_train)
poly_f1 = f1_score(y_train, poly_pred_train, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100), '\n')

rbf_accuracy = accuracy_score(y_train, rbf_pred_train)
rbf_f1 = f1_score(y_train, rbf_pred_train, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100), '\n')

linear_accuracy = accuracy_score(y_train, linear_pred_train)
linear_f1 = f1_score(y_train, linear_pred_train, average='weighted')
print('Accuracy (Linear Kernel): ', "%.2f" % (linear_accuracy*100))
print('F1 (Linear Kernel): ', "%.2f" % (linear_f1*100))


# In[9]:


# get the separating hyperplane
w = linear.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (linear.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the support vectors
b = linear.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = linear.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# In[12]:

# plot data
# sns.lmplot('Features', 'Clas', data=test, hue='Clas', palette='Set1', fit_reg=False, scatter_kws={"s": 70});
# plt.plot(xx, yy, linewidth=2, color='black')
# plt.plot(xx, yy_down, 'k--')
# plt.plot(xx, yy_up, 'k--')

