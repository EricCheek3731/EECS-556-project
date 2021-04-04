function Xhat=nlm_filter2D(Y,P,S,sigma_h)
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %  Y   : noisy image
 %  P   : size of patch
 %  S    : size of search window
 %  sigma_h: w(i,j) = exp(-||p(i) - p(j)||_2^2/(2*P^2*sigma_h^2))
 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 %% Parameters Computation
 [m, n]=size(Y);
 pixels = Y(:);
 s = m*n;
 
 half_P = floor(P/2);
 half_S = floor(S/2);

 %% Compute patches
 padInput = padarray(Y,[half_P half_P],'symmetric'); 
 patches = im2col(padInput, [P P], 'sliding')';
 
 %% Compute Pixel Pairs within the same Search Window
 indexes = reshape(1:s, m, n);
 padIndexes = padarray(indexes, [half_S half_S]);
 neighbors = im2col(padIndexes, [S S], 'sliding');
 TT = repmat(1:s, [S^2 1]);
 edges = [TT(:) neighbors(:)];
 %RR = find(TT(:) >= neighbors(:));
 %edges(RR, :) = [];
 edges(TT(:) >= neighbors(:),:) =[];
 
 %% Compute Weight Matrix using Weighted Euclidean distance
 diff = patches(edges(:,1), :) - patches(edges(:,2), :);
 V = exp(-sum(diff.*diff,2)/(2*P^2*sigma_h^2)); 
 W = sparse(edges(:,1), edges(:,2), V, s, s);
 
 %% Form Matrix and Set Diagonal Elements
 W = W + W' +spdiags(ones(s,1), 0, s, s);

 %% Normalize Weights
 W = spdiags(1./sum(W,2), 0, s, s)*W;
 
 %% Compute Denoised Image
 Xhat = W*pixels;
 Xhat = reshape(Xhat, m , n);