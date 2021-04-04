clc
close all;

%% Read Test Images
pathForImages ='';
% imageName = 'barbara.png';
% imageName = 'peppers.png';
% imageName = 'lena.png'
imageName = 'boat.png';
[X,pp]=imread(strcat([pathForImages,imageName])); % pp not needed here
X=im2double(X);
if (length(size(X))>2)
    X = rgb2gray(X);
end
if (max(X(:))<2)
    X = X*255;
end

%% Generate Noisy Image
sigma = 20;
% sigma = 30;
% sigma = 40;
% sigma = 50;
Y=X+sigma*randn(size(X));
%% Input PSNR Computation
PSNRIn = 20*log10(255/sqrt(mean((Y(:)-X(:)).^2)));

%% Setting Denoising parameters
P = 7;
S = 11;
%sigma_h = 15;
sigma_h = 0.55*sigma

%% Non-local Means Filter
tic
Xhat = nlm_filter2D(Y,P,S,sigma_h);
cpuTime=toc
%% Output PSNR Computation
PSNROut = 20*log10(255/sqrt(mean((Xhat(:)-X(:)).^2)));
%% Image Display
figure('Name','Original Clean Image'),imshow(X,[],'border','tight')
figure('Name',strcat(['Noisy image, ',num2str(PSNRIn),'dB'])),imshow(Y,[],'border','tight')
figure('Name',strcat(['Clean Image by Adaptive dictionary, ',num2str(PSNROut),'dB'])),imshow(Xhat,[],'border','tight')
figure('Name', 'Residual'),imshow(Y-Xhat,[],'border','tight')

%% Print PSNR
fprintf('PSNRIn=%f\n', PSNRIn);
fprintf('PSNROut=%f\n', PSNROut);