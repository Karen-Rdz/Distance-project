#Generates and saves a random signal with custom channels, frequency and seconds
#Signal is saved as csv file
#Next step is to load signal in time, with a filter and then calculate power

import pause
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []
y7 = []
y8 = [] 

freq = 256          #frequency in Hz
ms = (1/freq)*1000  #time in milliseconds
secs = 1           #seconds of data to generate
data = secs*freq    #data points in seconds

for k in range(data):              #k is int
  x.append(k)                    #x is list
  y1.append(np.random.randn(1))  #ys is list
  y2.append(np.random.randn(1))  #ys is list
  y3.append(np.random.randn(1))  #ys is list
  y4.append(np.random.randn(1))  #ys is list
  y5.append(np.random.randn(1))  #ys is list
  y6.append(np.random.randn(1))  #ys is list
  y7.append(np.random.randn(1))  #ys is list
  y8.append(np.random.randn(1))  #ys is list
  #print(type (x))
  #print(type(ys))
  #print(x)
  #print(ys)
  print (k)
  #pause.milliseconds(ms) #pause to plot, not needed
  pause.milliseconds(100) #100 milliseconds pause


# For one column
##df = pd.DataFrame(ys)
##df.to_csv('Sig1.csv', index=False)
##print(ys)

#For two or more columns
df = pd.DataFrame(data={"# Sample": x, "Ch1": y1, "Ch2": y2, "Ch3": y3, "Ch4": y4, "Ch5": y5, "Ch6": y6, "Ch7": y7, "Ch8": y8,})
df.to_csv("./Sigs8.csv", sep=',',index=False)
#if index = True, saves index values as well


##list1 = [1,2,3,4,5]
##df = pd.DataFrame(list1)
##df.to_csv('filename.csv', index=False)
##print(list1)
###index =false removes unnecessary indexing/numbering in the csv

plt.figure(1)
plt.subplot(811)
plt.plot(x,y1)

plt.subplot(812)
plt.plot(x,y2)

plt.subplot(813)
plt.plot(x,y3)

plt.subplot(814)
plt.plot(x,y4)

plt.subplot(815)
plt.plot(x,y5)

plt.subplot(816)
plt.plot(x,y6)

plt.subplot(817)
plt.plot(x,y7)

plt.subplot(818)
plt.plot(x,y8)

plt.show()



