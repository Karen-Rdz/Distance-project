#Simulate EEG Signals acquisition in real time
#Need to implement PSD calculation on one-second windows ("See Process.m")

import random
import pause
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Cch1 = []
Cch2 = []
#... Cch3, Ch4, ...
Cch8 = []
xx = []

for k in range(100): 

    #if file2 = 1, starts stream
    file2 = open("myfile2.txt","w")
    file2.write("1")
    file2.close() #to change file access modes

    #x Data 
    file1 = open("myfile.txt","w")
    L = str(k)
    file1.writelines(L)
    file1.close() #to change file access modes
    xx.append(k)
    #print(k)

    #y Data 
    file4 = open("myfile4.txt","w")
    y = np.random.uniform(-1,1,size = (1,8)) #outputs numpy.ndarray of size 1,8

    #Separate Channels
    ch1 = y[0][0]
    ch2 = y[0][1]
    ch3 = y[0][2]
    ch4 = y[0][3]
    ch5 = y[0][4]
    ch6 = y[0][5]
    ch7 = y[0][6]
    ch8 = y[0][7]

    #Append variables to vectors
    Cch1.append(ch1)
    Cch2.append(ch2)
    #... Cch3, Ch4, ...
    Cch8.append(ch8)

    #Concatenate channels as strings sepparated with commas
    Lch= str(ch1) + ',' + str(ch2) + ',' + str(ch3) + ',' + str(ch4) + ',' + str(ch5) + ',' + str(ch6) + ',' + str(ch7) + ',' + str(ch8)
    file4.writelines(Lch)
    file4.close() #to change file access modes
    #print(y)
    #print(ch1) #for channel 1,
    #print(type(y))

    #Counter
    file3 = open("myfile3.txt","w")
    L = str(k)
    file3.writelines(L)
    file3.close() #to change file access modes
    print(k)

    #pause.milliseconds(1/256) #Real time acquisition pause at 256 Hz
    pause.milliseconds(10) #100 milliseconds pause

#if file2 = 0, ends stream
file2 = open("myfile2.txt","w")
file2.write("0")
file2.close() #to change file access modes

print (Cch8)

#Plot Generated Signalss
plt.figure(1)
plt.subplot(811)
plt.plot(xx,Cch1)

plt.subplot(812)
plt.plot(xx,Cch2)

#... Cch3, Ch4, ...

plt.subplot(818)
plt.plot(xx,Cch8)
plt.tight_layout()
plt.show()




###################### Other Random Options ############################

    #y = random.randint(-1, 1) #data is int
    #y = random.random()  #random between 0 and 1
    #y = random.uniform(-1,1)
    #y = random.uniform(-1,1)  #random uniform between -1 and 1
    #y = np.random.randn(1) #data is nmpy array
