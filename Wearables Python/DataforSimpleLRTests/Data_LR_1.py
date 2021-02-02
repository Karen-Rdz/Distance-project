import scipy.io
import matplotlib.pyplot as plt
import csv 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Open score file
score = []
with open('score.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        for i in row:
            score.append(int(i))

# 8 channels
channels=['FP2','FP1','C4','C3','P8','P7','O1','O2']

PRC = scipy.io.loadmat('PRC.mat')   #PowRatios Codes
PowRatio = 10       #See PowRatioCodes
Channel = 4         #Use codes from above
#Try different Channels or PowRatios

sreal=[1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17] #Subject number
x = []

for k in range(0, len(sreal)):
    if score[k]<=21:
        col='g'     # No fatigue
    if score[k]>21 and score[k]<35:
        col='y'   # Normal fatigue
    if score[k]>=35:
        col='r'     #Extreme fatigue

    # Open file of the subject
    file1 = 'S'+str(sreal[k])+'_powratios6.mat'  
    Pow_ratios = scipy.io.loadmat(file1)

    # Add value of x
    x.append(Pow_ratios['Pow_ratios'][PowRatio-1, Channel-1])

    # Plot the value
    plt.plot(x[k],score[k], col+'o')
    
# y is the variable we want to predict
y = score

# Labels of the plot
xlab = str(PRC['PRC'][0, PowRatio-1]) + ' (' + str(channels[Channel-1]) + ')'
plt.xlabel(xlab)
plt.ylabel('FAS Score') # Fatigue Score

# plt.show()

# Fatigue classification according to FAS: 
clas = []
for i in range(len(y)):
    if y[i] <= 21:
        clas.append(1)         # No fatigue   
    elif y[i] >= 35:
        clas.append(3)         # Extreme Fatigue 
    else:
        clas.append(2)         # Normal fatigue

# You can try some Machine Learning codes to predict and compare
# classifications (Use Data - 70% for Training, 30% for Testing)

# ----- REGRESION LINEAL -----------

# Reshape x array to be one column
X = np.array(x).reshape((-1, 1))

# Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

# Create a linear regression to fit x and y train
modelo_regresion = LinearRegression().fit(X_train, y_train)

# Predict all the values of y
y_pred_all = modelo_regresion.predict(X)

# Predict the test values
y_pred_test = modelo_regresion.predict(X_test)

# Plot the linear regression line of all the data
plt.plot(X, y_pred_all, color = "blue", linewidth = 3)

# Plot the linear regression line of the test data
plt.plot(X_test, y_pred_test, color = "black", linewidth = 3)

# Show the plot
plt.show()

# Fatigue classification according to FAS with predicted values: 
clas_predicted = []
for i in range(len(y_pred_all)):
    if y_pred_all[i] <= 21:
        clas_predicted.append(1)      # No fatigue   
    elif y_pred_all[i] >= 35:
        clas_predicted.append(3)      # Extreme Fatig
    elif y_pred_all[i] > 21 and y_pred_all[i] < 35:     
        clas_predicted.append(2)    # Normal fatigue

# Display de original and predicted classification
print(clas)
print("-----------")
print(clas_predicted)

# Find the number of errors
error = 0
for i in range(len(clas)):
    if clas[i] != clas_predicted[i]:
        error += 1

print("Número de casos incorrectos: ", error)
print("Procentaje de casos incorrectos: ", (error*100)/17, "\n")


# ------- TRYING SVMs ---------

# Choose one method
clf_svc = svm.SVC(kernel='linear', C=1)
# clf_svc = svm.LinearSVC()
# clf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
# clf_svc = svm.SVC(kernel='poly', degree=3, gamma='auto', C=1.0)

# Fit the data and predict the values
clf_svc.fit(X_train, y_train)
y_pred_svc = clf_svc.predict(X_test)
y_pred_all_svc = clf_svc.predict(X)

# Fatigue classification according to FAS with predicted values: 
clas_predicted_svc = []
for i in range(len(y_pred_all_svc)):
    if y_pred_all_svc[i] <= 21:
        clas_predicted_svc.append(1)      # No fatigue   
    elif y_pred_all_svc[i] >= 35:
        clas_predicted_svc.append(3)      # Extreme Fatig
    elif y_pred_all_svc[i] > 21 and y_pred_all_svc[i] < 35:     
        clas_predicted_svc.append(2)      # Normal fatigue

print( "TRYING SVM")
# print(y)                # Mostrar valores de y 
# print(y_pred_all_svc)   # Mostrar que si cambian los valores
# Display the original and predicted classification using SVM
print(clas)
print("-----------")
print(clas_predicted_svc)

# Find the number of errors using SVM
error_svc = 0
for i in range(len(clas)):
    if clas[i] != clas_predicted_svc[i]:
        error_svc += 1

print("Número de casos incorrectos: ", error_svc)
print("Procentaje de casos incorrectos: ", (error_svc*100)/17)

# Trying to plot
# Y = np.array(y).reshape((-1, 1))
# h = .01

# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = Y[:, 0].min() - 1, Y[:, 0].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# Z = clf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
# Plot scatter x, y, Z

# Cross validation
pred = cross_val_predict(clf_svc, X, y, cv = 2)

# Fatigue classification according to FAS with predicted values: 
clas_predicted_valid = []
for i in range(len(pred)):
    if pred[i] <= 21:
        clas_predicted_valid.append(1)      # No fatigue   
    elif pred[i] >= 35:
        clas_predicted_valid.append(3)      # Extreme Fatig
    elif pred[i] > 21 and pred[i] < 35:     
        clas_predicted_valid.append(2)      # Normal fatigue

print( "TRYING CROSS VALIDATION")
# print(y)                # Mostrar valores de y 
# print(y_pred_all_svc)   # Mostrar que si cambian los valores
# Display the original and predicted classification using SVM
print(clas)
print("-----------")
print(clas_predicted_valid)

# Find the number of errors using SVM
error_valid = 0
for i in range(len(clas)):
    if clas[i] != clas_predicted_valid[i]:
        error_valid += 1

print("Número de casos incorrectos: ", error_valid)
print("Procentaje de casos incorrectos: ", (error_valid*100)/17)

