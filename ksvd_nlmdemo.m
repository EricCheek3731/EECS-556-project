%============================================================
%               demo - ksvd_nlm
%   denoising using ksvd with non-local means regularization
%   with function denoiseImageKSVD_nlm.m
%============================================================

clear
bb=8; % block size
RR=4; % redundancy factor
K=RR*bb^2; % number of atoms in the dictionary

%% Read Test Images
pathForImages ='./img/';
imageName = 'barbara.png';
% imageName = 'peppers.png';
% imageName = 'lena.png'
% imageName = 'boat.png';
[X,pp]=imread(strcat([pathForImages,imageName]));
X=im2double(X);
if (length(size(X))>2)
    X = rgb2gray(X);
end
if (max(X(:))<2)
    X = X*255;
end

%% Generate Noisy Image
% sigma = 20;
% sigma = 30;
% sigma = 40;
sigma = 50;
Y=X+sigma*randn(size(X));
%% Input PSNR Computation
PSNRIn = 20*log10(255/sqrt(mean((Y(:)-X(:)).^2)));

%% KSVD method
[Xhat,output] = denoiseImageKSVD_nlm(Y, sigma,K);
%% Output PSNR Computation
PSNROut = 20*log10(255/sqrt(mean((Xhat(:)-X(:)).^2)));

%% Image Display
figure('Name','Original clean image'),imshow(X,[],'border','tight')
figure('Name',strcat(['Noisy image, ',num2str(PSNRIn),'dB'])),imshow(Y,[],'border', 'tight')
figure('Name',strcat(['Clean Image by Adaptive dictionary, ',num2str(PSNROut),'dB'])),imshow(Xhat,[],'border','tight')

figure('Name','The dictionary trained on patches from the noisy image'),
I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb);

%% Print PSNR
fprintf('PSNRIn=%f\n', PSNRIn);
fprintf('PSNROut=%f\n', PSNROut);