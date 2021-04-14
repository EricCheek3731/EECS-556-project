%============================================================
%               demo - classify blocks on an image
%============================================================


bb=8; % block size

%% Read Test Images
pathForImages ='';
% imageName = 'barbara.png';
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
% sigma = 30;
% sigma = 40;
% sigma = 50;
Y=X+sigma*randn(size(X));

% % blocks as columns of matrix
% % blocks with stride 1(overlapping)
% % 1 One version % fast
% thrVar = 600; % properthreshold ?
% thrTex = 0.55; % properthreshold
% idx = [1:prod(size(Y)-bb+1)];
% cMat = zeros(size(Y)-bb+1);
% idxMat = ones(size(Y)-bb+1);
% [rows,cols] = ind2sub(size(idxMat),idx);
% for i = 1:length(idx)
%     currBlock = Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
%     currVar = var(currBlock(:)); % var computes expirical variance
% %    currVar = (1/(bb^2))* sum((currBlock(:) - mean(currBlock(:))).^2);
%     if currVar < thrVar % smooth
%         cMat(rows(i), cols(i)) = 1;
%     else
%         dx = conv2([1 -1], currBlock);
%         dy = conv2([1; -1], currBlock);
%         dxy = [dx(:) dy(:)];
%         lamda = svd(dxy);
%         r_q = lamda(1)/(lamda(1) + lamda(2));
%         if r_q < thrTex% texture
%            cMat(rows(i), cols(i)) = 2; 
%         else % edge
%             cMat(rows(i), cols(i)) = 3;
%         end
%     end
% end
% cMat
% size(find(cMat == 1)) % test if threshold is proper
% size(find(cMat == 2))
% size(find(cMat == 3))

% Test
% use for testing if the classification is reasonable
% idx_c1 = find(cMat == 1);
% idx_c2 = find(cMat == 2);
% idx_c3 = find(cMat == 3);
% i = idx_c1(1);
% figure('Name',strcat(['Image block belongs to class', num2str(cMat(rows(i), cols(i)))])),imshow(Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1),[],'border','tight')
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

% test 2 non-overlapping
thrVar = 600; % properthreshold ?
thrTex = 0.55; % properthreshold
cMat = zeros(size(Y)-bb+1);
idxMat = zeros(size(Y)-bb+1);
idxMat([[1:bb:end-1],end],[[1:bb:end-1],end]) = 1;
idx = find(idxMat);
[rows,cols] = ind2sub(size(idxMat),idx);
for i = 1:length(idx)
    currBlock = Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
    currVar = var(currBlock(:)); % var computes expirical variance
%    currVar = (1/(bb^2))* sum((currBlock(:) - mean(currBlock(:))).^2);
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

Yc = zeros(size(Y));
for i = 1:length(idx)
    if cMat(rows(i), cols(i)) == 2
        Yc(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = 255*ones(bb, bb);
    else
      if cMat(rows(i), cols(i)) == 3
        Yc(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1) = 127*ones(bb, bb);
      end
    end
end
 figure('Name','Yc'),imshow(Yc,[],'border','tight')       