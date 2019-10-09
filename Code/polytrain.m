% f = polytrain(X,Y,M,l) Train a polynomial to map f(X) = Y
%
% In:
%   X: Nx1 matrix, N 1D data points rowwise.
%   Y: Nx1 matrix, N 1D data points rowwise.
%   M: degree of the polynomial plus 1.
%   l: nonnegative scalar, the regularisation parameter. Default: 0.
% Out:
%   f: (struct) the polynomial, with fields:
%      type='polyf', coefficients w (Mx1), regularisation parameter l.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function f = polytrain(X,Y,M,l)

N = size(X,1); f.type = 'polyf'; f.l = l;

XX = bsxfun(@power,X,0:(M-1));
f.w = (XX'*XX + spdiags(repmat(l,M,1),0,M,M)) \ (XX'*Y);

