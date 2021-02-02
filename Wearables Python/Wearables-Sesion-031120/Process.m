%Simulate process of  real-time PSD calculation in EEG signals 
clearvars; close all; clc

Nch = 8;             %Number of Channels
Fs = 256;            %Sampling frequency
secs = 5;            %Seconds
Sigsize = Fs*secs;   %Signal size
Ws = Fs*1;           %Window size (in seconds*Fs)
cont = 0;            %Sample Counter
Wcont = 0;           %Window Counter

%Frequency bands (delta, theta, alpha, beta and gamma)
Freqbands = [1 4; 4 7; 8 12; 13 30; 30 50];

%%%%%%%%%%%%%%%%%%%%%%% BANDPASS FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%
%4th order 1-100 bandpass filter
%Cutoffs 1-100, 256 Sampling Freq

Lc = 1;   %Low cutoff frequency
Hc = 100; %High cutoff frequency
o = 4;    %Filter order

[a,b] = butter(o,[1 100]/(Fs/2));
d = designfilt('bandpassiir','FilterOrder',o, ...
    'HalfPowerFrequency1',Lc,'HalfPowerFrequency2',Hc, ...
    'SampleRate',Fs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic %Used to measure time
for k = 1:Sigsize
    cont = cont+1; 
    
    %Random 8-channel signal
    y(k,:) = randn(1,Nch); %y(k,:)
    Yw(cont,:) = y(k,:);
    
    %Comment pause for faster simulation
    pause(1/Fs); %Sampling Frequency pause to simulate real-time acquisition
    
    %Reset counter every time the window is full and perform calculations:
    if cont == Ws; 
        
    Wcont = Wcont + 1; %Increase Window Counter
     
    %Applying bandpass filter (0.1-100 Hz)  
    yf = filter(a,b,Yw);  %A Banstop filter at 60 Hz is also needed
    
    % Squaring the signal
    yfsq = yf.*yf; 
     
    %Applying Fourier Transform
    fta = abs(fft(yfsq)); 
    
    %Power Spectrum Density Calculation
    PSD = log(fta(1:fix(length(fta)/2),:));
    
    %Mean of PSD during each window across 8 channels (5 freq bands)
    Delta(Wcont,:) = mean(PSD(Freqbands(1,1):Freqbands(1,2),:)); 
    Theta(Wcont,:) = mean(PSD(Freqbands(2,1):Freqbands(2,2),:));
    Alpha(Wcont,:) = mean(PSD(Freqbands(3,1):Freqbands(3,2),:));
    Beta(Wcont,:) =  mean(PSD(Freqbands(4,1):Freqbands(4,2),:));
    Gamma(Wcont,:) = mean(PSD(Freqbands(5,1):Freqbands(5,2),:));
    
        cont = 0; 
    end 
        
end 
toc %Used to measure time

%%

%%%%%%%%%%%%%%%%% Raw Signals Visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; suptitle ('Raw Signals in Time');
subplot(811);plot(y(:,1)); axis('tight');
subplot(812);plot(y(:,2)); axis('tight');
subplot(813);plot(y(:,3)); axis('tight');
%...
subplot(818);plot(y(:,8)); axis('tight');
xlabel('Sample')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% Filtered Signals Visualization %%%%%%%%%%%%%%%%%%%%%%
figure; suptitle ('Filtered Signals (One Second)');
subplot(811);plot(yf(:,1)); axis('tight');
subplot(812);plot(yf(:,2)); axis('tight');
subplot(813);plot(yf(:,3)); axis('tight');
%...
subplot(818);plot(yf(:,8)); axis('tight');
xlabel('Sample')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% Squared Filtered Signals  %%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; suptitle ('Squared Filtered Signals (One Second)')
subplot(811);plot(yfsq(:,1)); axis('tight');
subplot(812);plot(yfsq(:,2)); axis('tight');
subplot(813);plot(yfsq(:,3)); axis('tight');
%...
subplot(818);plot(yfsq(:,8)); axis('tight');
xlabel('Sample')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% Power Spectrum Density %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; suptitle ('PSD in 1-(Fs/2) Hz')
subplot(811);plot(PSD(:,1)); axis('tight');
subplot(812);plot(PSD(:,2)); axis('tight');
subplot(813);plot(PSD(:,3)); axis('tight');
%...
subplot(818);plot(PSD(:,8)); axis('tight');
xlabel('Frequency (Hz)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% Plot PSD across 8 channels in time %%%%%%%%%%%%%%%%%%%%%
figure; suptitle ('PSD in time')
subplot(511); plot(Delta); axis('tight');
subplot(512); plot(Theta); axis('tight');
subplot(513); plot(Alpha); axis('tight');
subplot(514); plot(Beta);  axis('tight');
subplot(515); plot(Gamma); axis('tight');
xlabel ('Time (Seconds)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


