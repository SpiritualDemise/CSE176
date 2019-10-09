% f = rbftrain(X,Y,M,s,l) Train RBF to map f(X) = Y
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   Y: NxD matrix, N D-dim data points rowwise.
%   M: number of RBF centres.
%   s: nonnegative scalar, the RBF width.
%   l: nonnegative scalar, the regularisation parameter. Default: 0.
% Out:
%   f: (struct) the RBF, with fields:
%      type='rbf', centres C (MxL), width s, regularisation parameter l,
%      weights W (DxM), biases w (Dx1).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
 
function f = rbftrain(X,Y,M,s,l)

N = size(X,1); f.type = 'rbf'; f.s = s; f.l = l;

f.C = X(randperm(N,M),:);	% centres

% Fit weights
G = exp(-sqdist(f.C,X)/(2*s*s));
G1 = sum(G,2); GG = G*G' - G1*(G1'/N);
GG = GG + spdiags(repmat(l,M,1),0,M,M);
f.W = (Y'*G'-mean(Y',2)*G1') / GG; f.w = mean(Y'-f.W*G,2);

