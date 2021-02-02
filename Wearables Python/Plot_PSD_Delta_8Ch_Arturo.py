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
    y = lfilter(b, a, data, axis=0)
    return y
##############################################

#Stores PSD values in time     (1x8 vectors)
Delta = np.zeros((1,8))        #Delta band (0.1-4 Hz)
Theta = np.zeros((1,8))        #Theta band (4-7 Hz)
Alpha = np.zeros((1,8))        #Alpha band (8-12 Hz)
Beta  = np.zeros((1,8))        #Beta  band (13-30 Hz)
Gamma = np.zeros((1,8))        #Gamma band (30-50 Hz)
#print(Delta)
#Frequencies for delta (d), theta (t), alpha (a), beta (b) and gamma (g)
d1 = 1; d2 = 4; t1 = 4; t2 = 7; a1 = 8; a2 = 12; b1 = 13; b2 = 30; g1 = 30; g2 = 50; 


CH=np.zeros((1,8))

Fs = 256            #Sampling Frequency
secs = 150           #Seconds of signal
sigsize = secs*Fs   #Signal Size

w1 = np.zeros((Fs,8))            #Window for ch1

lowcut = 0.1        #Lowcut Frecuency (Change to 0.1 Hz)
highcut = 100   #Highcut Frequency (Change to 100 Hz)

for k in range(sigsize): 

    #y Data 
    y = np.random.uniform(-1,1,size = (1,8)) #outputs numpy.ndarray of size 1,8
    CH=np.vstack((CH,y)) #Stores all y values
    
    if k%Fs!=0 or k==0:   #if the residuum of k/Fs is 0 (when k = Fs or is multiple)
        w1[k%Fs,:]=y      #Append ch1 window
    
    
    #Calculations to perform on every window
    else:     #When window is full, perform filter and PSD calculation

        #print(w1)       #Print ch1 window
        #Bandpass filter
        y1 = butter_bandpass_filter(w1, lowcut, highcut, Fs, order=4)
        #Squared signal
        ysq = y1*y1
        #Fourier Transform
        yf = scipy.fftpack.fft(ysq,axis=0)
        #Power Spectral Density
        yf = np.abs(yf)       
        yf = np.log(yf)

        #yf outputs 256x8 matrix (window data x channels)
        #print(np.size(yf,0))
        #print(np.size(yf,1))

        
        #Append PSD values in five frequency bands every "second"
        Delta=np.vstack((Delta,np.mean(yf[d1:d2,:],axis=0)))
        Theta=np.vstack((Theta,np.mean(yf[t1:t2,:],axis=0)))
        Alpha=np.vstack((Alpha,np.mean(yf[a1:a2,:],axis=0)))
        Beta=np.vstack((Beta,np.mean(yf[d1:d2,:],axis=0)))
        Gamma=np.vstack((Gamma,np.mean(yf[d1:d2,:],axis=0)))

        #Stores values in matrices of size (Wxch), W is the window counter, ch is 8 channels
        #print(np.size(Delta,0))
        #print(np.size(Delta,1))
        #print(Delta)
        
        ##### Un comment to see the following plots
        ###Raw signal
##        plt.figure(k//Fs)
##        plt.subplot(311)
##        plt.plot(np.arange(np.size(w1)/8),w1[:,1]) #Just one channel
##      ###Filtered signal
##        plt.subplot(312)
##        plt.plot(np.arange(np.size(w1)/8),y1[:,1])
##      ###Fourier Transform
##        plt.subplot(313)
##        
##        plt.plot(np.arange(np.size(w1)/8),yf[:,1])
        ################################################
        

    #pause.milliseconds(1/Fs) #Real time acquisition pause at 256 Hz
    #pause.milliseconds(10) #100 milliseconds pause

#Stores mean PSD values in five frequency bands every second, for 8 channels
CH = np.delete(CH, (0), axis=0)
Delta = np.delete(Delta, (0), axis=0)
Theta = np.delete(Theta, (0), axis=0)
Alpha= np.delete(Alpha, (0), axis=0)
Beta= np.delete(Beta, (0), axis=0)
Gamma= np.delete(Gamma, (0), axis=0)

#Stores all y values in matrix of size (SamplexCh)
#print(np.size(CH,0))
#print(np.size(CH,1))


#Print offline Delta PSD vs time (seconds)
#plt.close('all')
#plt.figure(1)
#plt.plot(np.arange(np.size(Delta)/8),Delta[:,0])
#plt.show()


#Plot Generated Signalss
plt.figure(1)
plt.subplot(811)
plt.plot(np.arange(np.size(Delta)/8),Delta[:,0])

plt.subplot(812)
plt.plot(np.arange(np.size(Delta)/8),Delta[:,1])

#... Cch3, Ch4, ...

plt.subplot(818)
plt.plot(np.arange(np.size(Delta)/8),Delta[:,7])
plt.tight_layout()
plt.show()



###################### Other Random Options ############################

    #y = random.randint(-1, 1) #data is int
    #y = random.random()  #random between 0 and 1
    #y = random.uniform(-1,1)
    #y = random.uniform(-1,1)  #random uniform between -1 and 1
    #y = np.random.randn(1) #data is nmpy array

