%===========================================================
%                   demo - olap_ksvdc
%   denoising using k-svd on classified overlapping patches
%                with function olapKSVD.m
%===========================================================
clear
bb = 8; % block size
RR = 1; % redundancy factor
K = RR*bb^2; % number of atoms in the dictionary

% Define an overlap and a corresponding step_size
overlap = 5;
step_size = bb - overlap;

%% Read Test Images
pathForImages = './img/';
imageName = 'boat.png';
[X,pp] = imread(strcat([pathForImages,imageName]));
X = im2double(X);
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
Y = X + sigma * randn(size(X));

%% KSVD method
K = K - step_size;
[Xhat,ouput] = olapKSVD(Y, sigma, K, bb, step_size);

%% Input PSNR Computation and Output PSNR Computation
PSNRIn = 20*log10(255/sqrt(mean((Y(:)-X(:)).^2)));
PSNROut = 20*log10(255/sqrt(mean((Xhat(:)-X(:)).^2)));

%% Image Display
figure('Name','Original Image'),imshow(X,[],'border','tight')
figure('Name',strcat(['Noisy Image, ',num2str(PSNRIn),'dB'])),imshow(Y,[],'border', 'tight')
figure('Name',strcat(['Denoised Image, ',num2str(PSNROut),'dB'])),imshow(Xhat,[],'border','tight')

%% Print PSNR
fprintf('PSNRIn=%f\n', PSNRIn);
fprintf('PSNROut=%f\n', PSNROut);