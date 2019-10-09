% Y = linf(X,f) Value of linear function y = f(x) = W.x+w
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   f: (struct) the linear function, with fields:
%      type='linf', W (DxL), w (Dx1).
% Out:
%   Y: NxD matrix, N D-dim outputs Y = f(X).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function Y = linf(X,f)

Y = bsxfun(@plus,X*f.W',f.w');

