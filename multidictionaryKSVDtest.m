function [IOut,output] = denoiseImageKSVD(Image,sigma,K,varargin)
%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   A  D I C T  I O N A R Y
%                  T R A I N E D   O N   N O I S Y   I M A G E
%==========================================================================
% function IOut = denoiseImageKSVD(Image,sigma,K,varargin)
% denoise an image by sparsely representing each block with the
% already overcomplete trained Dictionary, and averaging the represented parts.
% Detailed description can be found in "Image Denoising Via Sparse and Redundant
% representations over Learned Dictionaries", (appeared in the 
% IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).
% This function may take some time to process. Possible factor that effect
% the processing time are:
%  1. number of KSVD iterations - the default number of iterations is 10.
%  However, fewer iterations may, in most cases, result an acceleration in
%  the process, without effecting  the result too much. Therefore, when
%  required, this parameter may be re-set.
%  2. maxBlocksToConsider - The maximal number of blocks to train on. If this 
%  number is larger the number of blocks in the image, random blocks
%  from the image will be selected for training. 
% ===================================================================
% INPUT ARGUMENTS : Image - the noisy image (gray-level scale)
%                   sigma - the s.d. of the noise (assume to be white Gaussian).
%                   K - the number of atoms in the trained dictionary.
%    Optional arguments:              
%                  'blockSize' - the size of the blocks the algorithm
%                       works. All blocks are squares, therefore the given
%                       parameter should be one number (width or height).
%                       Default value: 8.
%                       'errorFactor' - a factor that multiplies sigma in order
%                       to set the allowed representation error. In the
%                       experiments presented in the paper, it was set to 1.15
%                       (which is also the default  value here).
%                  'maxBlocksToConsider' - maximal number of blocks that
%                       can be processed. This number is dependent on the memory
%                       capabilities of the machine, and performances�
%                       considerations. If the number of available blocks in the
%                       image is larger than 'maxBlocksToConsider', the sliding
%                       distance between the blocks increases. The default value
%                       is: 250000.
%                  'slidingFactor' - the sliding distance between processed
%                       blocks. Default value is 1. However, if the image is
%                       large, this number increases automatically (because of
%                       memory requirements). Larger values result faster
%                       performances (because of fewer processed blocks).
%                  'numKSVDIters' - the number of KSVD iterations processed
%                       blocks from the noisy image. If the number of
%                       blocks in the image is larger than this number,
%                       random blocks from all available blocks will be
%                       selected. The default value for this parameter is:
%                       10 if sigma > 5, and 5 otherwise.
%                  'maxNumBlocksToTrainOn' - the maximal number of blocks
%                       to train on. The default value for this parameter is
%                       65000. However, it might not be enough for very large
%                       images
%                  'displayFlag' - if this flag is switched on,
%                       announcement after finishing each iteration will appear,
%                       as also a measure concerning the progress of the
%                       algorithm (the average number of required coefficients
%                       for representation). The default value is 1 (on).
%                  'waitBarOn' - can be set to either 1 or 0. If
%                       waitBarOn==1 a waitbar, presenting the progress of the
%                       algorithm will be displayed.
% OUTPUT ARGUMENTS : Iout - a 2-dimensional array in the same size of the
%                       input image, that contains the cleaned image.
%                    output.D - the trained dictionary.
% =========================================================================

% first, train a dictionary on the noisy image

reduceDC = 1;
[NN1,NN2] = size(Image);
waitBarOn = 1;
if (sigma > 5)
    numIterOfKsvd = 10;
else
    numIterOfKsvd = 5;
end
C = 1.15;%  'errorFactor' - a factor that multiplies sigma in order to set the allowed representation error.
maxBlocksToConsider = 260000;
slidingDis = 1;
bb = 8;% 'blockSize' - the size of the blocks the algorithm works. All blocks are squares( bb = width or height).
maxNumBlocksToTrainOn = 65000;
displayFlag = 1;

for argI = 1:2:length(varargin)
    if (strcmp(varargin{argI}, 'slidingFactor'))
        slidingDis = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'errorFactor'))
        C = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'maxBlocksToConsider'))
        maxBlocksToConsider = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'numKSVDIters'))
        numIterOfKsvd = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'blockSize'))
        bb = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'maxNumBlocksToTrainOn'))
        maxNumBlocksToTrainOn = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'displayFlag'))
        displayFlag = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'waitBarOn'))
        waitBarOn = varargin{argI+1};
    end
end

% % first, train a dictionary on blocks from the noisy image
% % get blkMatrix
% if(prod([NN1,NN2]-bb+1)> maxNumBlocksToTrainOn) % if blocks > maxNumBlocksToTrainOn
%     randPermutation =  randperm(prod([NN1,NN2]-bb+1)); % select blocks randomly
%     selectedBlocks = randPermutation(1:maxNumBlocksToTrainOn);
% 
%     blkMatrix = zeros(bb^2,maxNumBlocksToTrainOn);% n x k
%     for i = 1:maxNumBlocksToTrainOn
%         [row,col] = ind2sub(size(Image)-bb+1,selectedBlocks(i)); % corresponding left top row,col of each block
%         currBlock = Image(row:row+bb-1,col:col+bb-1);% each block
%         blkMatrix(:,i) = currBlock(:);% each column of blkMatrix is a block
%     end
% else
%     blkMatrix = im2col(Image,[bb,bb],'sliding'); % if blocks < maxNumBlocksToTrainOn use im2col directly
% end

% comment out previous code for simple test
% thrVar = 300; % propertheshold ?
% idx = [1:prod(size(Y)-bb+1)];
% cMat = zeros(size(Y)-bb+1);
% idxMat = ones(size(Y)-bb+1);
% [rows,cols] = ind2sub(size(idxMat),idx);
% blkMatrix = im2col(Image,[bb,bb],'sliding');
% for i = 1:length(idx)
%     currBlock = Y(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
%     currVar = sum((currBlock(:) - mean(currBlock(:))).^2);
%     if currVar < thrVar % smooth
%         cMat(rows(i), cols(i)) = 1;
%     end
% end

thrVar = 600; % properthreshold ?
thrTex = 0.55; % properthreshold
idx = [1:prod(size(Image)-bb+1)];
cMat = zeros(size(Image)-bb+1);
idxMat = ones(size(Image)-bb+1);
[rows,cols] = ind2sub(size(idxMat),idx);
blkMatrix = im2col(Image,[bb,bb],'sliding');
for i = 1:length(idx)
    currBlock = Image(rows(i):rows(i)+bb-1,cols(i):cols(i)+bb-1);
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

% blkMatrix
idxc1 = find(cMat == 1);
blkMatrixc1 = blkMatrix(:, idxc1);
idxc2 = find(cMat == 2);
blkMatrixc2 = blkMatrix(:, idxc2);
idxc3 = find(cMat == 3);
blkMatrixc3 = blkMatrix(:, idxc3);

% param are parmeters passed into KSVD function
% param contains K, numIteration, errorFlage, errorGoal, preserveDCAtom,
% initialDictionary, InitializationMethod, waitBarHandle,
% counterForWaitBar,displayProgress
% all not influence by using mutiple dictionary
param.K = K;
param.numIteration = numIterOfKsvd ;

param.errorFlag = 1; % decompose signals until a certain error is reached. do not use fix number of coefficients.
param.errorGoal = sigma*C;
param.preserveDCAtom = 0;

% get initialization DCT Dictionary
Pn=ceil(sqrt(K));% ceil computes nearest integer larger than or equal to the number. K is the number of atoms in the trained dictionary
DCT=zeros(bb,Pn);
for k=0:1:Pn-1
    V=cos([0:1:bb-1]'*k*pi/Pn);
    if k>0
        V=V-mean(V);
    end
    DCT(:,k+1)=V/norm(V);
end
DCT=kron(DCT,DCT); % Overcomplete DCT

param.initialDictionary = DCT(:,1:param.K );
param.InitializationMethod =  'GivenMatrix';

% Preprocessing blkMatrix
if (reduceDC) % reduceDC
%     vecOfMeans = mean(blkMatrix);
%     blkMatrix = blkMatrix-ones(size(blkMatrix,1),1)*vecOfMeans;
    vecOfMeans1 = mean(blkMatrixc1);
    blkMatrixc1 = blkMatrixc1-ones(size(blkMatrixc1,1),1)*vecOfMeans1;
    vecOfMeans2 = mean(blkMatrixc2);
    blkMatrixc2 = blkMatrixc2-ones(size(blkMatrixc2,1),1)*vecOfMeans2;
    vecOfMeans3 = mean(blkMatrixc3);
    blkMatrixc3 = blkMatrixc3-ones(size(blkMatrixc3,1),1)*vecOfMeans3;
end

if (waitBarOn)
    counterForWaitBar = param.numIteration+1;
    h = waitbar(0,'Denoising In Process ...'); % a small window with bar showing the progress
    param.waitBarHandle = h;
    param.counterForWaitBar = counterForWaitBar;
end

% When having multiple dictionary use KSVD on each category
% blkMatrix is used for getting dictionary by KSVD
param.displayProgress = displayFlag;
% [Dictionary,output] = KSVD(blkMatrix,param);
% output.D = Dictionary;
[Dictionary1, ~] = KSVD(blkMatrixc1, param);
[Dictionary2, ~] = KSVD(blkMatrixc2, param);
[Dictionary3, ~] = KSVD(blkMatrixc3, param);
output.D1 = Dictionary1;
output.D2 = Dictionary2;
output.D3 = Dictionary3;

if (displayFlag)
    disp('finished Trainning dictionary');
end


% denoise the image using the resulted dictionary
errT = sigma*C;
IMout=zeros(NN1,NN2);
Weight=zeros(NN1,NN2);
%blocks = im2col(Image,[NN1,NN2],[bb,bb],'sliding');

% % decide slidingDis based on maxBlocksToConsider
% while (prod(floor((size(Image)-bb)/slidingDis)+1)>maxBlocksToConsider)
%     slidingDis = slidingDis+1;
% end
% % get blocks for OMP?
% [blocks,idx] = my_im2col(Image,[bb,bb],slidingDis);
% 
% if (waitBarOn)
%     newCounterForWaitBar = (param.numIteration+1)*size(blocks,2);
% end
if (waitBarOn)
    newCounterForWaitBar1 = (param.numIteration+1)*size(blkMatrixc1,2);
    newCounterForWaitBar2 = (param.numIteration+1)*size(blkMatrixc2,2);
    newCounterForWaitBar3 = (param.numIteration+1)*size(blkMatrixc3,2);
end

% separate block into blocks in different category and combine them
% together after processing ?
% go with jumps of 30000
% for jj = 1:30000:size(blocks,2) % ? reduce DC and do OMPerr batch by batch
%     if (waitBarOn)
%         waitbar(((param.numIteration*size(blocks,2))+jj)/newCounterForWaitBar);
%     end
%     jumpSize = min(jj+30000-1,size(blocks,2)); % often size(blocks, 2)
%     
%     if (reduceDC)
%         vecOfMeans = mean(blocks(:,jj:jumpSize));
%         blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1); % reduce DC for each block
%     end
%     % get updated sparse representation
%     Coefs = OMPerr(Dictionary,blocks(:,jj:jumpSize),errT);
%      
%     % blocks here are denoised block
%     if (reduceDC)
%         blocks(:,jj:jumpSize)= Dictionary*Coefs + ones(size(blocks,1),1) * vecOfMeans; % add DC back for each block
%     else
%         blocks(:,jj:jumpSize)= Dictionary*Coefs;
%     end
% end

for jj = 1:30000:size(blkMatrixc1,2) % ? reduce DC and do OMPerr batch by batch
    if (waitBarOn)
        waitbar(((param.numIteration*size(blkMatrixc1,2))+jj)/newCounterForWaitBar1);
    end
    jumpSize = min(jj+30000-1,size(blkMatrixc1,2)); % often size(blocks, 2)
    
    if (reduceDC)
        vecOfMeans = mean(blkMatrixc1(:,jj:jumpSize));
        blkMatrixc1(:,jj:jumpSize) = blkMatrixc1(:,jj:jumpSize) - repmat(vecOfMeans,size(blkMatrixc1,1),1); % reduce DC for each block
    end
    % get updated sparse representation
    Coefs = OMPerr(Dictionary1,blkMatrixc1(:,jj:jumpSize),errT);
     
    % blocks here are denoised block
    if (reduceDC)
        blkMatrixc1(:,jj:jumpSize)= Dictionary1*Coefs + ones(size(blkMatrixc1,1),1) * vecOfMeans; % add DC back for each block
    else
        blkMatrixc1(:,jj:jumpSize)= Dictionary1*Coefs;
    end
end

for jj = 1:30000:size(blkMatrixc2,2) % ? reduce DC and do OMPerr batch by batch
    if (waitBarOn)
        waitbar(((param.numIteration*size(blkMatrixc2,2))+jj)/newCounterForWaitBar2);
    end
    jumpSize = min(jj+30000-1,size(blkMatrixc2,2)); % often size(blocks, 2)
    
    if (reduceDC)
        vecOfMeans = mean(blkMatrixc2(:,jj:jumpSize));
        blkMatrixc2(:,jj:jumpSize) = blkMatrixc2(:,jj:jumpSize) - repmat(vecOfMeans,size(blkMatrixc2,1),1); % reduce DC for each block
    end
    % get updated sparse representation
    Coefs = OMPerr(Dictionary2,blkMatrixc2(:,jj:jumpSize),errT);
     
    % blocks here are denoised block
    if (reduceDC)
        blkMatrixc2(:,jj:jumpSize)= Dictionary2*Coefs + ones(size(blkMatrixc2,1),1) * vecOfMeans; % add DC back for each block
    else
        blkMatrixc2(:,jj:jumpSize)= Dictionary2*Coefs;
    end
end

for jj = 1:30000:size(blkMatrixc3,2) % ? reduce DC and do OMPerr batch by batch
    if (waitBarOn)
        waitbar(((param.numIteration*size(blkMatrixc3,2))+jj)/newCounterForWaitBar3);
    end
    jumpSize = min(jj+30000-1,size(blkMatrixc3,2)); % often size(blocks, 2)
    
    if (reduceDC)
        vecOfMeans = mean(blkMatrixc3(:,jj:jumpSize));
        blkMatrixc3(:,jj:jumpSize) = blkMatrixc3(:,jj:jumpSize) - repmat(vecOfMeans,size(blkMatrixc3,1),1); % reduce DC for each block
    end
    % get updated sparse representation
    Coefs = OMPerr(Dictionary3,blkMatrixc3(:,jj:jumpSize),errT);
     
    % blocks here are denoised block
    if (reduceDC)
        blkMatrixc3(:,jj:jumpSize)= Dictionary3*Coefs + ones(size(blkMatrixc3,1),1) * vecOfMeans; % add DC back for each block
    else
        blkMatrixc3(:,jj:jumpSize)= Dictionary3*Coefs;
    end
end

% count = 1;
count1 = 1;
count2 = 1;
count3 = 1;
Weight = zeros(NN1,NN2);
IMout = zeros(NN1,NN2);
[rows,cols] = ind2sub(size(Image)-bb+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);
    % block is denoised block size bb x bb
    % blocks is denoised blocks matrix bb^2 x count
%     block =reshape(blocks(:,count),[bb,bb]);
    if cMat(row, col) == 1
        block = reshape(blkMatrixc1(:,count1), [bb, bb]);
        count1 = count1+1;
    else
        if  cMat(row, col) == 2
            block = reshape(blkMatrixc2(:,count2), [bb, bb]);
            count2 = count2+1;
        else
            block = reshape(blkMatrixc3(:,count3), [bb, bb]);
            count3 = count3+1;
        end
    end
    % Compute denoised blcok and weight block by block
    IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block;
    Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
%     count = count+1; % count is idx for blocks matrix. row, col is location for block on image
end

if (waitBarOn)
    close(h);
end
IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight); %?

