% Y = polyf(X,f) Value of 1D polynomial f(x)
%
% f(x) = w1 + w2.x + w3.x^2 + ... + wM.x^(M-1)
%
% In:
%   X: Nx1 matrix, N 1D data points rowwise.
%   f: (struct) the polynomial, with fields:
%      type='polyf', coefficients w (Mx1), regularisation parameter l.
% Out:
%   Y: Nx1 matrix, N 1D outputs Y = f(X).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function Y = polyf(X,f)

Y = bsxfun(@power,X,0:(length(f.w)-1))*f.w;

