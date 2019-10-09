% Y = rbf(X,f) Value of RBF network f(x)
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   f: (struct) the RBF, with fields:
%      type='rbf', centres C (MxL), width s, weights W (DxM), biases w (Dx1),
%      regularisation parameter l.
% Out:
%   Y: NxD matrix, N D-dim outputs Y = f(X).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function Y = rbf(X,f)

phi = exp(-sqdist(f.C,X)/(2*f.s*f.s));
Y = (bsxfun(@plus,f.W*phi,f.w))';

