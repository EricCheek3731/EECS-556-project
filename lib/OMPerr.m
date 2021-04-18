function [A]=OMPerr(D,X,errorGoal) 
%=============================================
% Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: D - the dictionary
%                  X - the signals to represent
%                  errorGoal - the maximal allowed representation error for
%                  each siganl.
% output arguments: A - sparse coefficient matrix.
%=============================================
[n,P]=size(X);
E2 = errorGoal^2*n; % n*(sigma*C)^2
maxNumCoef = n/2;
A = sparse(size(D,2),size(X,2)); % sparse matrix with all zeros
for k=1:1:P % loop for each blocks
    x=X(:,k);
    residual=x;
	indx = [];
	a = [];
	currResNorm2 = sum(residual.^2);
	j = 0;
    while currResNorm2>E2 && j < maxNumCoef
		j = j+1;
        proj=D'*residual; % D'(Da) = sum( sigma^2 v v_T) * a. sigma - singluar value
        pos=find(abs(proj)==max(abs(proj)));  % find the position of the  max singular value
        pos=pos(1); % only the row number is useful.
        indx(j)=pos; % add it to indx
        a=pinv(D(:,indx(1:j)))*x; % calculate sparse representation
        residual=x-D(:,indx(1:j))*a; % calculate residual                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
		currResNorm2 = sum(residual.^2);
   end
   if (~isempty(indx))
       A(indx,k)=a;
   end
end
return;
