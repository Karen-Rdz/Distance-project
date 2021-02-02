#Simulate EEG Signals acquisition in real time
#Stores signals in a window of size Fs and once it is full, performs:
#Bandpass filter, and PSD calculation every "second"
#Plots Delta PSD in one channel
#Need to implement the same but in all 8 channels, as well as power ratios
#Need to try with OpenBCI


import random
import pause
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.signal import butter, lfilter

###### FILTERS ############################
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
##############################################

#These values store the whole signals
Cch1 = []
Cch2 = []
#... Cch3, Ch4, ...
Cch8 = []
xx = []
Wx = []
#############################################
#Stores PSD values in time
Delta = []         #Delta band (0.1-4 Hz)
Theta = []         #Theta band (4-7 Hz)
Alpha = []         #Alpha band (8-12 Hz)
Beta  = []         #Beta  band (13-30 Hz)
Gamma = []         #Gamma band (30-50 Hz)
#Frequencies for delta (d), theta (t), alpha (a), beta (b) and gamma (g)
d1 = 1; d2 = 4; t1 = 4; t2 = 7; a1 = 8; a2 = 12; b1 = 13; b2 = 30; g1 = 30; g2 = 50; 

Wcont = 0           #Window counter
cont = 0            #General counter

Fs = 256            #Sampling Frequency
secs = 20           #Seconds of signal
sigsize = secs*Fs   #Signal Size

w1 = []             #Window for ch1
x1 = []             #X for Windows
lowcut = 0.1        #Lowcut Frecuency (Change to 0.1 Hz)
highcut = 100       #Highcut Frequency (Change to 100 Hz)

for k in range(sigsize): 

    #x Data 
    xx.append(k)
    #print(k)

    #y Data 
    y = np.random.uniform(-1,1,size = (1,8)) #outputs numpy.ndarray of size 1,8
    #print(y)
    #print(ch1) #for channel 1,
    #print(type(y))
    
    #Separate Channels
    ch1 = y[0][0]
    ch2 = y[0][1]
    ch3 = y[0][2]
    ch4 = y[0][3]
    ch5 = y[0][4]
    ch6 = y[0][5]
    ch7 = y[0][6]
    ch8 = y[0][7]

    #Append variables to vectors (Stores all values from each channel)
    Cch1.append(ch1)
    Cch2.append(ch2)
    #... Cch3, Ch4, ...
    Cch8.append(ch8)

    cont = cont +1      #Increase counter
    w1.append(ch1)      #Append ch1 window
    x1.append(cont)     #Append x of window
    
    #print (cont)
    
    #Calculations to perform on every window
    if cont == Fs:      #When window is full, perform filter and PSD calculation
        Wcont = Wcont+1 #Increase window counter
        print(',')
        print (Wcont)   #Print window counter
        print(',')
        cont = 0        #Reset counter
        #print(w1)       #Print ch1 window
        #Bandpass filter
        y1 = butter_bandpass_filter(w1, lowcut, highcut, Fs, order=4)
        #Squared signal
        ysq = y1 * y1
        #Fourier Transform
        yf = scipy.fftpack.fft(ysq)
        #Power Spectral Density
        yf = np.abs(yf)       
        yf = np.log(yf)
        
        #Append PSD values in five frequency bands every "second"
        Delta.append(np.mean(yf[d1:d2]))
        #Theta.append(np.mean(yf[t1:t2]))
        #Alpha.append(np.mean(yf[a1:a2]))
        #Beta.append(np.mean(yf[b1:b2]))
        #Gamma.append(np.mean(yf[g1:g2]))
        
        Wx.append(Wcont)

        
        ##### Un comment to see the following plots
        ###Raw signal
##        plt.figure(1)
##        plt.subplot(311)
##        plt.plot(x1,w1)
##      ###Filtered signal
##        plt.subplot(312)
##        plt.plot(x1,y1)
##      ###Fourier Transform
##        plt.subplot(313)
##        plt.plot(x1,yf)
##        plt.show()
        ################################################
        
        w1 = []         #Erase ch1 window to refill
        x1 = []         #Erase x1 window to refill

    pause.milliseconds(1/Fs) #Real time acquisition pause at 256 Hz
    #pause.milliseconds(10) #100 milliseconds pause



print(Delta)
print(Wx)
#print (Cch8)

#Print offline Delta PSD vs time (seconds)
plt.figure(2)
plt.plot(Wx,Delta)
plt.show()


#Plot Generated Signalss
##plt.figure(1)
##plt.subplot(811)
##plt.plot(xx,Cch1)
##
##plt.subplot(812)
##plt.plot(xx,Cch2)
##
###... Cch3, Ch4, ...
##
##plt.subplot(818)
##plt.plot(xx,Cch8)
##plt.tight_layout()
##plt.show()
##


###################### Other Random Options ############################

    #y = random.randint(-1, 1) #data is int
    #y = random.random()  #random between 0 and 1
    #y = random.uniform(-1,1)
    #y = random.uniform(-1,1)  #random uniform between -1 and 1
    #y = np.random.randn(1) #data is nmpy array
