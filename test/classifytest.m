%============================================================
%               demo - classifytest
%           classify blocks on an image
%============================================================

bb=8; % block size

%% Read Test Images
pathForImages ='';
%imageName = 'barbara.png';
% imageName = 'peppers.png';
imageName = 'lena.png';
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
sigma = 20;
% sigma = 40;
% sigma = 40;
% sigma = 50;
Y=X+sigma*randn(size(X));

% non-overlapping
thrVar = (sigma/10 - 1) * 650; % proper threshold
thrTex = 0.96 + (sigma/10 - 1) * 0.01;
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
        %         dx = conv2([1 -1], currBlock);
        %         dy = conv2([1; -1], currBlock);
        % get filter length
        filterExtent = ceil(4*sigma);
        x = -filterExtent:filterExtent;
        
        % gaussian kernal
        c = 1/(sqrt(2*pi)*sigma);
        gaussKernel = c * exp(-(x.^2)/(2*sigma^2));
        gaussKernel = gaussKernel/sum(gaussKernel);
        
        % gaussian filter
        aSmooth=imfilter(currBlock,gaussKernel,'conv','replicate');   
        aSmooth=imfilter(aSmooth,gaussKernel','conv','replicate'); 
        
        % gradient func(1d deriv of Gaussian kernal)
        derivGaussKernel = gradient(gaussKernel);
        
        % normalize
        negVals = derivGaussKernel < 0;
        posVals = derivGaussKernel > 0;
        derivGaussKernel(posVals) = derivGaussKernel(posVals)/sum(derivGaussKernel(posVals));
        derivGaussKernel(negVals) = derivGaussKernel(negVals)/abs(sum(derivGaussKernel(negVals)));
        
        % gradient
        dx = imfilter(aSmooth, derivGaussKernel, 'conv','replicate');
        dy = imfilter(aSmooth, derivGaussKernel', 'conv','replicate');
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

Yc = zeros(size(Y));
for i = 1:length(idx)
    if cMat(rows(i), cols(i)) == 2 % texture white
        Yc(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = 255*ones(bb, bb);
    elseif cMat(rows(i), cols(i)) == 3 % edge gray
        Yc(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = 127*ones(bb, bb);
    else
        Yc(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = 63*ones(bb, bb);
    end
end
figure('Name','Original clean image'),imshow(X,[],'border','tight')
figure('Name','Yc'),imshow(Yc,[],'border','tight')

% show each image
% Yc1 = zeros(size(Y));
% Yc2 = zeros(size(Y));
% Yc3 = zeros(size(Y));
% for i = 1:length(idx)
%     if cMat(rows(i), cols(i)) == 2
%         Yc2(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = 255*ones(bb, bb);
%     else
%       if cMat(rows(i), cols(i)) == 3
%         Yc3(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = 255*ones(bb, bb);
%       else
%         Yc1(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = 255*ones(bb, bb);
%       end
%     end
% end
% figure('Name','Yc1'),imshow(Yc1,[],'border','tight')
% figure('Name','Yc2'),imshow(Yc2,[],'border','tight')
% figure('Name','Yc3'),imshow(Yc3,[],'border','tight')