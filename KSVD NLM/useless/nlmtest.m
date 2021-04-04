close all;
clear;
clc;
%% Original Image
X = imread('peppers.png');
figure, imshow(X), colormap gray
[M, N] = size(X);
%% Noisy Image
Y = imnoise(X, 'gaussian',0, 0.01);%?
disp(max(X(:)));
% nsigma = 20;
% randn('seed', 212096)
% Y = double(X) + nsigma*randn(M, N);
% Y = max(0,min(Y, 255));
figure, imshow(Y), colormap gray
%% MSE PSNR Computation
Y_MSE = sum(sum((double(Y)-double(X)).^2))/(M*N);
Y_PSNR = 10*log10(255^2/Y_MSE);
fprintf('Y_PSNR=%f\n', Y_PSNR);
%% Non-local Means Filter
Xhat = imnlmfilt(Y,'SearchWindowSize',21,'ComparisonWindowSize',7);
figure, imshow(Xhat), colormap gray
%% MSE PSNR Computation
Xhat_MSE = sum(sum((double(Xhat)-double(X)).^2))/(M*N);
Xhat_PSNR = 10*log10(255^2/Xhat_MSE);
fprintf('Xhat_PNSR=%f\n', Xhat_PSNR);
