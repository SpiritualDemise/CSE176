% [Y,J] = mlp(X,f) Value of multilayer perceptron f(x) = V.s(W.x+w)+v
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   f: (struct) the MLP, with fields:
%      type='mlp'; weight matrices (incl. bias) layers 1/2: W1 (Hx(L+1)),
%      W2 (Dx(H+1)); weight decay parameter l.
% Out:
%   Y: NxD matrix, N D-dim outputs Y = f(X).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function Y = mlp(X,f)

N = size(X,1);

[s,ds] = sigmoid(f.W1*[X';ones(1,N)]);
Y = (f.W2*[s;ones(1,N)])';

