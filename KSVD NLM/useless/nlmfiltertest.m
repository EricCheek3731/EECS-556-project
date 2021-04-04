close all;
clear;
clc;

%% Non-local means Filter Parameters
nsize = 7;% Neighbor Window Size
ssize = 21;% Search Window Size
hsigma = 5;% Sigma for Gaussian Kernel Generation

%% Image Input
X = imread('lena.png');

if(size(X,3)==3)
    X = rgb2gray(X);
end
[M, N] = size(X);

%% Gaussian Noise Addition
nsigma = 20;
randn('seed', 212096)
Y = double(X) + nsigma*randn(M, N);
Y = max(0,min(Y, 255));

%% MSE PSNR Computation
Y_MSE = sum(sum((double(Y)-double(X)).^2))/(M*N);
Y_PNSR = 10*log10(255^2/Y_MSE);

%% Noise Level Estimation using Robust Median Estimator 
dchw = dchwtf2(Y, 1);%?
tt = dchw{1}(:)';%?
std_dev2 = median(abs(tt))/0.6745;%?
%% Non-local means Filter
Xhat = nlm_filt(Y, hsigma, nsize, ksize, std_dev2);
%% MSE PSNR Computation
Xhat_MSE = sum(sum((double(Xhat)-double(X)).^2))/(M*N);
Xhat_PNSR = 10*log10(255^2/Xhat_MSE);
%% Denoised Image Plot
figure, imshow(uint(Y)), colormap gray