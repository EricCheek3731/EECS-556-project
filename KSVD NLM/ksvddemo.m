%============================================================
%               demo2 - denoise an image
% this is a run_file the demonstrate how to denoise an image, 
% using dictionaries. The methods implemented here are the same
% one as described in "Image Denoising Via Sparse and Redundant
% representations over Learned Dictionaries", (appeared in the 
% IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).
%============================================================

clear
bb=8; % block size
RR=4; % redundancy factor
K=RR*bb^2; % number of atoms in the dictionary

%% Read Test Images
pathForImages ='';
% imageName = 'barbara.png';
% imageName = 'peppers.png';
% imageName = 'lena.png'
imageName = 'boat.png';
[X,pp]=imread(strcat([pathForImages,imageName]));
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

%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   A   D I C T  I O N A R Y
%                  T R A I N E D   O N   N O I S Y   I M A G E
%==========================================================================
%% KSVD method
[Xhat,output] = denoiseImageKSVD(Y, sigma,K);
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