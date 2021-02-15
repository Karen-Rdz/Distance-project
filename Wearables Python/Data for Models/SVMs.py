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
from sklearn.preprocessing import StandardScaler 

# Get data
data_complete = scipy.io.loadmat('FeatureMatrices.mat')
labels = data_complete['FMinfo']
test = data_complete['WFM_1']

x = np.delete(test, 129, axis=1)
y = test[:, 129]

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

        # Standardization of data
        scaler = StandardScaler().fit(X_new)
        X_scaled = scaler.transform(X_new)

        # Split the data for train and test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3) 

        # RBF, Polynomial and Linear Kernel with standardization
        rbf = svm.SVC(kernel='rbf', gamma=0.6, C=1).fit(X_train, y_train)
        poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
        linear = svm.SVC(kernel='linear').fit(X_train, y_train)

        # Predict values
        poly_pred_test = cross_val_predict(poly, X_test, y_test, cv = 5)
        rbf_pred_test = cross_val_predict(rbf, X_test, y_test, cv = 5)
        linear_pred_test = cross_val_predict(linear, X_test, y_test, cv = 5)

        # Accuracy test
        poly_accuracy = accuracy_score(y_test, poly_pred_test)
        rbf_accuracy = accuracy_score(y_test, rbf_pred_test)
        linear_accuracy = accuracy_score(y_test, linear_pred_test)

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
print('----- Final results -----')
print(dict_features)

# 2 best features
X_new = SelectKBest(chi2, k=2).fit_transform(abs(x), y)
best_features = []
count = 0

# Know which are the best features
for i in X_new[0]:
    for j in abs(x[0]):
        count += 1
        if i == j:
            best_features.append(count)
            count = 0
            break
print(best_features)
print('--------')            

# Standardization of data
scaler = preprocessing.StandardScaler().fit(X_new)
X_scaled = scaler.transform(X_new)

# Plot data
plt.plot(X_new, "ob")
plt.title('Before Standardization')
plt.show()

plt.plot(X_scaled, "ob")
plt.title('After Standardization')
plt.show()

# Plot 2D
col = []
for i in range(0, len(y)):
    if y[i] == 1:
        col.append('g')
    elif y[i] == 2:
        col.append('y')
    elif y[i] == 3:
        col.append('r')
    plt.plot(X_scaled[i, 1], X_scaled[i, 0], col[i]+"o")
plt.title('2 features 2D')
plt.xlabel('X Scaled 129')
plt.ylabel('X Scaled 33')
plt.show()

# Plot 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(0, len(y)):
    ax.scatter3D(X_scaled[i, 1], X_scaled[i, 0], y[i], c=col[i], marker = 'o')
plt.title('2 features 3D')
ax.set_xlabel('X Scaled 129')
ax.set_ylabel('X Scaled 33')
ax.set_zlabel('Y clas')
plt.show()
