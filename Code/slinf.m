% Y = slinf(X,f) Value of logistic regression function y = f(x) = s(W'.x+w)
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   f: (struct) the logistic regression function, with fields:
%      type='slinf', W (1xL), w (scalar).
% Out:
%   Y: Nx1 matrix, N outputs in [0,1] Y = f(X).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function Y = slinf(X,f)

Y = sigmoid(X*f.W'+f.w);

