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
%   P E R F O R M   D E N O I S I N G   U S I N G   T H R E E   D I C T  I O N
%   A R I E S
%                  T R A I N E D   O N   N O I S Y   I M A G E
%==========================================================================
thrVar = 600; % properthreshold ?
thrTex = 0.55; % properthreshold
cMat = zeros(size(Y)-bb+1);
idxMat = zeros(size(Y)-bb+1);
idxMat([[1:bb:end-1],end],[[1:bb:end-1],end]) = 1;
idx = find(idxMat);
[rows,cols] = ind2sub(size(idxMat),idx);
for i = 1:length(idx)
    currBlock = Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
    currVar = (1/(bb^2))* sum((currBlock(:) - mean(currBlock(:))).^2);
    if currVar < thrVar % smooth
        cMat(rows(i), cols(i)) = 1;
    else
        dx = conv2([1 -1], currBlock);
        dy = conv2([1; -1], currBlock);
        dxy = [dx(:) dy(:)];
        lamda = svd(dxy);
        r_q = lamda(1)/(lamda(1) + lamda(2));
        if r_q < thrTex% texture
           cMat(rows(i), cols(i)) = 2; 
        else % edge
            cMat(rows(i), cols(i)) = 3;
        end
    end
end

Yc1 = zeros(size(Y));
Yc2 = zeros(size(Y));
Yc3 = zeros(size(Y));
for i = 1:length(idx)
    if cMat(rows(i), cols(i)) == 2 % texture white
        Yc2(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
    else
      if cMat(rows(i), cols(i)) == 3 % edge grey
        Yc3(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
      else
        Yc1(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);  
      end
    end
end

PSNRIn1 = 20*log10(255/sqrt(mean((Yc1(:)-Xc1(:)).^2)));
PSNRIn2 = 20*log10(255/sqrt(mean((Yc2(:)-Xc2(:)).^2)));
PSNRIn3 = 20*log10(255/sqrt(mean((Yc3(:)-Xc3(:)).^2)));

Xc1 = zeros(size(Y));
Xc2 = zeros(size(Y));
Xc3 = zeros(size(Y));
for i = 1:length(idx)
    if cMat(rows(i), cols(i)) == 2 % texture white
        Xc2(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = X(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
    else
      if cMat(rows(i), cols(i)) == 3 % edge grey
        Xc3(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = X(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
      else
        Xc1(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = X(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);  
      end
    end
end
%% KSVD method
[Xhat1,output1] = denoiseImageKSVD(Yc1, sigma,K);
[Xhat2,output2] = denoiseImageKSVD(Yc2, sigma,K);
[Xhat3,output3] = denoiseImageKSVD(Yc3, sigma,K);
Xhat = Xhat1 + Xhat2 + Xhat3;
% Xhat = zeros(size(Y));
% for i = 1:length(idx)
%     if cMat(rows(i), cols(i)) == 2 % texture white
%         Xhat(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = Xhat2(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
%     else
%       if cMat(rows(i), cols(i)) == 3 % edge grey
%         Xhat(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = Xhat3(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
%       else
%         Xhat(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = Xhat1(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);  
%       end
%     end
% end
%% Output PSNR Computation
PSNROut1 = 20*log10(255/sqrt(mean((Xhat1(:)-Xc1(:)).^2)));
PSNROut2 = 20*log10(255/sqrt(mean((Xhat2(:)-Xc2(:)).^2)));
PSNROut3 = 20*log10(255/sqrt(mean((Xhat3(:)-Xc3(:)).^2)));

PSNROut = 20*log10(255/sqrt(mean((Xhat(:)-X(:)).^2)));

%% Image Display
figure('Name','Original clean image'),imshow(X,[],'border','tight')
figure('Name',strcat(['Noisy image, ',num2str(PSNRIn),'dB'])),imshow(Y,[],'border', 'tight')
figure('Name',strcat(['Clean Image by Adaptive dictionary, ',num2str(PSNROut),'dB'])),imshow(Xhat,[],'border','tight')

figure('Name', strcat(['Clean Image c1, ',num2str(PSNROut1),'dB'])),imshow(Xhat3,[],'border','tight')
figure('Name', strcat(['Clean Image c2, ',num2str(PSNROut2),'dB'])),imshow(Xhat1,[],'border','tight')
figure('Name', strcat(['Clean Image c3, ',num2str(PSNROut3),'dB'])),imshow(Xhat2,[],'border','tight')

% figure('Name','The dictionary trained on patches from the noisy image'),
% I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb);

%% Print PSNR
fprintf('PSNRIn=%f\n', PSNRIn);
fprintf('PSNRIn1=%f\n', PSNRIn1);
fprintf('PSNRIn2=%f\n', PSNRIn2);
fprintf('PSNRIn3=%f\n', PSNRIn3);
fprintf('PSNROut=%f\n', PSNROut);
fprintf('PSNROut1=%f\n', PSNROut1);
fprintf('PSNROut2=%f\n', PSNROut2);
fprintf('PSNROut3=%f\n', PSNROut3);