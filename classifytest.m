%============================================================
%               demo - classify blocks on an image
%============================================================


bb=8; % block size

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

% blocks as columns of matrix
% 1 One version % fast
thrVar = 13; % propertheshold ?
idx = [1:prod(size(Y)-bb+1)];
cMat = zeros(size(Y)-bb+1);
idxMat = ones(size(Y)-bb+1);
[rows,cols] = ind2sub(size(idxMat),idx);
for i = 1:length(idx)
    currBlock = Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
%     currVar = var(currBlock(:)); % var computes expirical variance
    currVar = (1/(bb^2))* sum((currBlock(:) - mean(currBlock(:))).^2);
    if currVar < thrVar % smooth
        cMat(rows(i), cols(i)) = 1;
    end
end
cMat
size(find(cMat == 1)) % test if threshold is proper
% % 2 Another version % slow
% thrVar = 300; % propertheshold ?
% idx = [1:prod(size(Y)-bb+1)];
% cMat = zeros(size(Y)-bb+1);
% idxMat = ones(size(Y)-bb+1);
% [rows,cols] = ind2sub(size(idxMat),idx);
% blocks = im2col(Y, [bb bb], 'sliding');
% for i = 1:length(idx)
%     currBlock = blocks(:, idx);
%     currVar = sum((currBlock(:) - mean(currBlock(:))).^2);
%     if currVar < thrVar % smooth
%         cMat(rows(i), cols(i)) = 1;
%     end
% end
% cMat
