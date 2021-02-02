clearvars; close all; clc

load('score.mat');
      %    1    2     3    4    5    6    7    8
channels={'FP2','FP1','C4','C3','P8','P7','O1','O2'}; %EEG Channels

load('PRC.mat'); %PowRatios Codes
PowRatio = 10;    %See PowRatioCodes
Channel = 4;     %Use codes from above
%Try different Channels or PowRatios

sreal=[1 2 5 6 7 8 9 11 12 13 14 15 16 17]; %Subject number

%Load subjects Data
for k = 1:length(sreal)
    
    if score(k)<=21;col=[0 1 0];end;% No fatigue 
    if score(k)>21 && score(k)<35; col=[1 0.5 0];end% Normal fatigue 
    if score(k)>=35;col=[1 0 0];end;% Extreme fatigue

    file = strcat('S',num2str(sreal(k)),'_powratios6.mat');
    
    load (file);
    
    x(k) = Pow_ratios(PowRatio,Channel);
    
    plot(x(k),score(k),'ko','markersize',7,'markerfacecolor',col); hold on; 
         
end 

% y is the variable we want to predict
y = score; 

gcf; %Get current figure

xlab = strcat(PRC{PowRatio},' (',channels{Channel},')'  );
xlabel(xlab);
ylabel('FAS Score'); %Fatigue Score

%Fatigue classification according to FAS: 
clas = ones(size(y)).*2;         %Normal fatigue
c1 = find(y<=21); clas(c1) = 1;  %No fatigue
c3 = find(y>=35); clas (c3) = 3; %Extreme Fatigue

% You can try some Machine Learning codes to predict and compare
% classifications (Use Data - 70% for Training, 30% for Testing)
