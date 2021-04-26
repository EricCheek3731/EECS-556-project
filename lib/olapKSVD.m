function [IOut,output] = olapKSVD(Image, sigma, K, bb, step_size, varargin)
%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   A  D I C T  I O N A R Y
%                  T R A I N E D   O N   N O I S Y   I M A G E
%==========================================================================
% Denoise an image by K-SVD method with overlapping patches
%  1. number of KSVD iterations - the default number of iterations is 10.
%  2. maxBlocksToConsider - The maximal number of blocks to train on. 
%  If this number is larger the number of blocks in the image, random 
%  blocks from the image will be selected for training.
%  Detailed description can be found in "Image Denoising Via Sparse and 
%  Redundant representations over Learned Dictionaries", (appeared in the 
%  IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).
% =========================================================================
% INPUT ARGUMENTS : Image = the noisy image (gray-level scale)
%                   sigma = the s.d. of the noise (assume to be white Gaussian).
%                   K = the number of atoms in the trained dictionary.
%                   bb = the block size.
%                   step_size = the block size - the overlapping size.
% =========================================================================
%    Optional arguments:
%                  'errorFactor' - a factor that multiplies sigma in order
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

reduceDC = 1;
[NN1,NN2] = size(Image);
waitBarOn = 1;
if (sigma > 5)
    numIterOfKsvd = 10;
else
    numIterOfKsvd = 5;
end

C = 1.15;
maxBlocksToConsider = 260000;
slidingDis = 1;
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

if (sigma <= 5)
    numIterOfKsvd = 5;
end

%% Classify noisy image
idx = [1:prod(size(Image)-bb+1)];
cMat = zeros(size(Image)-bb+1);
idxMat = zeros(size(Image)-bb+1);
idxMat([[1:step_size:end-1],end], [[1:step_size:end-1],end]) = 1; % Overlapping
[rows,cols] = ind2sub(size(idxMat),idx);
blkMatrix = im2col(Image, [bb,bb], 'sliding');
for i = 1:length(idx)
    if i < length(idx)/3
        cMat(rows(i), cols(i)) = 1;
    else
        if i < 2*length(idx)/3
            cMat(rows(i), cols(i)) = 2; 
        else
            cMat(rows(i), cols(i)) = 3;
        end
    end
end

%% Train a dictionary on blocks from the noisy image

% Blocks used to train dict
blocks1 = blkMatrix(:, find(cMat == 1));
blocks2 = blkMatrix(:, find(cMat == 2));
blocks3 = blkMatrix(:, find(cMat == 3));

% The number of blocks <= maxNumBlocksToTrainOn
if(size(blocks1,2) > maxNumBlocksToTrainOn)
    blkMatrixc1 = zeros(bb^2, maxNumBlocksToTrainOn);
    for i = 1:maxNumBlocksToTrainOn
        blkMatrixc1(:,i) = blocks1(:,i);
    end
else
    blkMatrixc1 = blocks1;
end
if(size(blocks2,2) > maxNumBlocksToTrainOn) 
    blkMatrixc2 = zeros(bb^2, maxNumBlocksToTrainOn);
    for i = 1:maxNumBlocksToTrainOn
        blkMatrixc2(:,i) = blocks2(:,i);
    end
else
    blkMatrixc2 = blocks2;
end
if(size(blocks3,2) > maxNumBlocksToTrainOn) 
    blkMatrixc3 = zeros(bb^2, maxNumBlocksToTrainOn);
    for i = 1:maxNumBlocksToTrainOn
        blkMatrixc3(:,i) = blocks3(:,i);
    end
else
    blkMatrixc3 = blocks3;
end

param.K = K;
param.numIteration = numIterOfKsvd ;
param.errorFlag = 1;
param.errorGoal = sigma*C;
param.preserveDCAtom = 0;

%% Get initialization DCT Dictionary
Pn=ceil(sqrt(K));
DCT=zeros(bb,Pn);
for k=0:1:Pn-1
    V=cos([0:1:bb-1]'*k*pi/Pn);
    if k>0
        V=V-mean(V); 
    end
    DCT(:,k+1)=V/norm(V);
end
DCT=kron(DCT,DCT);

param.initialDictionary = DCT(:,1:param.K ); % use K columns
param.InitializationMethod =  'GivenMatrix';

if (reduceDC)
    blkMatrixc1 = blkMatrixc1 - ones(size(blkMatrixc1,1),1) * mean(blkMatrixc1);
    blkMatrixc2 = blkMatrixc2 - ones(size(blkMatrixc2,1),1) * mean(blkMatrixc2);
    blkMatrixc3 = blkMatrixc3 - ones(size(blkMatrixc3,1),1) * mean(blkMatrixc3);
end

if (waitBarOn)
    counterForWaitBar = param.numIteration+1;
    h = waitbar(0,'Denoising In Process ...');
    param.waitBarHandle = h;
    param.counterForWaitBar = counterForWaitBar;
end

param.displayProgress = displayFlag;
[Dictionary1,~] = KSVD(blkMatrixc1, param);
output.D1 = Dictionary1;
[Dictionary2,~] = KSVD(blkMatrixc2, param);
output.D2 = Dictionary2;
[Dictionary3,~] = KSVD(blkMatrixc3, param);
output.D3 = Dictionary3;

if (displayFlag)
    disp('finished Trainning dictionary');
end

%% Denoise the image using the trained dictionary
errT = sigma*C;

if (waitBarOn)
    newCounterForWaitBar1 = (param.numIteration+1) * size(blocks1,2);
    newCounterForWaitBar2 = (param.numIteration+1) * size(blocks2,2);
    newCounterForWaitBar3 = (param.numIteration+1) * size(blocks3,2);
end

% go with jumps of 30000
for jj = 1:30000:size(blocks1,2) % reduce DC and do OMPerr batch by batch
    if (waitBarOn)
        waitbar(((param.numIteration*size(blocks1,2))+jj)/newCounterForWaitBar1);
    end
    jumpSize = min(jj+30000-1,size(blocks1,2)); % often size(blocks, 2)
    
    if (reduceDC)
        vecOfMeans = mean(blocks1(:,jj:jumpSize));
        blocks1(:,jj:jumpSize) = blocks1(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks1,1),1); % reduce DC for each block
    end
    
    % get updated sparse representation
    Coefs = OMPerr(Dictionary1,blocks1(:,jj:jumpSize),errT);
     
    % blocks here are denoised block
    if (reduceDC)
        blocks1(:,jj:jumpSize) = Dictionary1*Coefs + ones(size(blocks1,1),1) * vecOfMeans; % add DC back for each block
    else
        blocks1(:,jj:jumpSize) = Dictionary1*Coefs;
    end
end

for jj = 1:30000:size(blocks2,2) % reduce DC and do OMPerr batch by batch
    if (waitBarOn)
        waitbar(((param.numIteration*size(blocks2,2))+jj)/newCounterForWaitBar2);
    end
    jumpSize = min(jj+30000-1,size(blocks2,2)); % often size(blocks, 2)
    
    if (reduceDC)
        vecOfMeans = mean(blocks2(:,jj:jumpSize));
        blocks2(:,jj:jumpSize) = blocks2(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks2,1),1); % reduce DC for each block
    end
    % get updated sparse representation
    Coefs = OMPerr(Dictionary2,blocks2(:,jj:jumpSize),errT);
     
    % blocks here are denoised block
    if (reduceDC)
        blocks2(:,jj:jumpSize) = Dictionary2*Coefs + ones(size(blocks2,1),1) * vecOfMeans; % add DC back for each block
    else
        blocks2(:,jj:jumpSize) = Dictionary2*Coefs;
    end
end

for jj = 1:30000:size(blocks3,2) % reduce DC and do OMPerr batch by batch
    if (waitBarOn)
        waitbar(((param.numIteration*size(blocks3,2))+jj)/newCounterForWaitBar3);
    end
    jumpSize = min(jj+30000-1,size(blocks3,2)); % often size(blocks, 2)
    
    if (reduceDC)
        vecOfMeans = mean(blocks3(:,jj:jumpSize));
        blocks3(:,jj:jumpSize) = blocks3(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks3,1),1); % reduce DC for each block
    end
    % get updated sparse representation
    Coefs = OMPerr(Dictionary3,blocks3(:,jj:jumpSize),errT);
     
    % blocks here are denoised block
    if (reduceDC)
        blocks3(:,jj:jumpSize) = Dictionary3*Coefs + ones(size(blocks3,1),1) * vecOfMeans; % add DC back for each block
    else
        blocks3(:,jj:jumpSize) = Dictionary3*Coefs;
    end
end

count1 = 1;
count2 = 1;
count3 = 1;
Weight = zeros(NN1,NN2);
IMout = zeros(NN1,NN2);
[rows,cols] = ind2sub(size(Image)-bb+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);
    if cMat(row, col) == 1
        block = reshape(blocks1(:,count1), [bb, bb]);
        count1 = count1+1;
    elseif  cMat(row, col) == 2
        block = reshape(blocks2(:,count2), [bb, bb]);
        count2 = count2+1;
    else
        block = reshape(blocks3(:,count3), [bb, bb]);
        count3 = count3+1;  
    end
    % Compute denoised blcok and weight block by block
    IMout(row:row+bb-1,col:col+bb-1) = IMout(row:row+bb-1,col:col+bb-1)+block;
    Weight(row:row+bb-1,col:col+bb-1) = Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
    % count is idx for blocks matrix. row, col is location for block on image
end

if (waitBarOn)
    close(h);
end
IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);
% IOut = (Image+0.000015*sigma*IMout)./(1+0.00005*sigma*Weight1 + 0.000004*sigma*Weight2 + 0.000015*sigma*Weight3);
end

