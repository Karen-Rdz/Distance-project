clearvars; close all; clc

Fs = 256;             %Sampling frequency
segs = 4;             %Seconds of signal
sigsize = Fs*segs;    %Signal size

x = randn(1,sigsize); %Random signal
t = [1:sigsize];      %Time vector

%%%%%%%%%%%%%%%%%%%%% Signal in time %%%%%%%%%%%%%%%%%%
figure;subplot(311);plot(t,x);
axis('tight');
xlabel ('Samples');
ylabel ('Amplitude (\muV)');

x1 = x(1:Fs*1);
hold on; plot(t(1:length(x1)),x1,'r');

%%%%%%%%%%%%%%%%%%% Square of signal %%%%%%%%%%%%%%%%%%
x1sq = x1.*x1;
subplot(312); plot(x1sq,'r','linewidth',1);
axis('tight');
xlabel ('Samples');
ylabel ('Amplitude^2 (\muV^2)')

%%%%%%%%%%%%%%%% Power Spectrum Density %%%%%%%%%%%%%%%
fta = abs(fft(x1sq));
%Plot logarithm of the real part of the FFT of one second of signal
subplot(313);plot(log(fta(1:fix(length(fta)/2))),'r','linewidth',1.5);

axis('tight');
xlabel('Frequency (Hz)');
ylabel('PSD');

%This needs to be implemented in continuous time to calculate every second


