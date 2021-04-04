function img = nlm_filt(Y, hsigma, nsize, ssize, nstd)
%% Compute half size
half_nsize = floor(nsize/2);
half_ssize = floor(ssize/2);

%% Take care of boundaries
[M, N] = size(Y);
Yo = zeros(M+ssize-1, N+ssize-1);
Yo(half_ssize+1:M+half_ssize, half_ssize+1:N+half_ssize) = Y;
Yo(1:half_ssize, :) = Yo(ssize:-1:half_ssize+2,:);
Yo(M+half_ssize:M+half_ssize-1,:) = Yo(M+half_ssize-1:-1:M,:);
Yo(:,1:half_ssize) = Yo(:, ssize:-1:half_ssize+2);
Yo(:,N+half_ssize:N+half_ssize-1) = Yo(:,N+half_ssize-1:-1:N);
%% Gaussian Kernel Generation
kernel = gauss_ker(hsigma, nsize);

%% Non-local Means Filter
filt_h = 0.55*nstd %?
[Mo, No] = size(Yo);
for i = half_ssize+1:M-half_ssize
    for j = half_ssize+1:N-half_ssize
        nwin = Yo(i-half_nsize:i+half_nsize, j-half_nsize:j+half_nsize);
        swin = Yo(i-half_ssize:i+half_ssize, j-half_ssize:j+half_ssize);
        for ii=1:(ssize-nsize+1)
            for jj=1:(ssize-nsize+1)
                edist = nwin-swin(ii:ii+nsize-1, jj+nsize-1);
                dist = sum(sum(kernel.*edist))/(nsize^2)
                weight(ii,jj)=exp(-max(dist-(2*nstd^2),0)/filtg^2);
            end
        end
        wsum = sum(sum(weight));
        nweight = weight/wsum;
        psum = sum(sum(swin(half_nsize+1:ssize-half_nsize, half_nsize+1:ssize-half_nsize)));
        img(i-half_ssize,j-half_ssize)=psum;
    end
end
end

