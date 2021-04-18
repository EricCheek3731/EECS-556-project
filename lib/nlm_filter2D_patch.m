function Xhat=nlm_filter2D_patch(Y,m,n,P,S,sigma)
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %  Y    : spare "a"
 %  m,n  : image size
 %  P    : size of patch
 %  S    : size of search window
 %  sigma: noise
 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 %% Parameters Computation
 s = m*n;
 sigma_h = 0.55*sigma;
 M = size(Y,2);
 %% Compute patches
 pixels = Y';
 patches = Y';
 
%% Compute Search Window
 TT = repmat(1:M, [S 1]);
 TT = [TT(:) TT(:)];
 for i = 1:size(TT,1)
     if mod(i,S) == 1
         x = TT(i,1);
     end

     if mod(i,S) == 0 
         TT(i,2) = x + mod(i,S) + S;
     else 
         TT(i,2) = x + mod(i,S);
     end
 end
 TT(TT(:,2) >= M, :) = [];
 edges = TT;
 %% Compute Weight Matrix using Weighted Euclidean distance
 diff = patches(edges(:,1), :) - patches(edges(:,2), :);
 V = exp(-sum(diff.*diff,2)/(2*P^2*sigma_h^2)); 
 W = sparse(edges(:,1), edges(:,2), V, M, M);
 
 %% Form Matrix and Set Diagonal Elements
 W = W + W' +spdiags(ones(s,1), 0, M, M);

 %% Normalize Weights
 W = spdiags(1./sum(W,2), 0, M, M)*W;
 
 %% Compute Denoised Image
 Xhat = (W*pixels)';