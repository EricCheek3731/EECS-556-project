function kernel = gauss_ker(hsigma, nsize)
x = [-floor(nsize/2):floor(nsize/2)];
x1 = repmat(x.^2, nsize, 1);
x2 = x1'+x1;
kernel=exp(-(x2/(2*(hsigma)^2));
end

