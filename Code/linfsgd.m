% [G,E] = linfsgd(X,Y,Xv,Yv,eta,maxit,B,g0)
% Train linear function y = f(x) = W.x+w with stochastic gradient descent
%
% In:
%   X: NxL matrix, N L-dim data points rowwise, training set inputs.
%   Y: NxD matrix, N D-dim data points rowwise, training set outputs.
%   Xv: MxL matrix, M L-dim data points rowwise, validation set inputs.
%   Yv: MxD matrix, M D-dim data points rowwise, validation set outputs.
%   eta: positive scalar, the learning rate (step size).
%   maxit: number of iterations to run.
%   B: minibatch size, between 1 and N.
%   g0: (struct) initial linear function (see definition in linf.m).
% Out:
%   G: (ceil(N/B))x(maxit+1) cell array where G{n,i} is a struct containing
%      the linear function at minibatch n and iteration i-1 (see struct
%      definition in linf.m).
%   E: 2x(ceil(N/B))x(maxit+1) array where E(1,n,i) and E(2,n,i) contain the
%      training and validation error at minibatch n and iteration i-1.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [G,E] = linfsgd(X,Y,Xv,Yv,eta,maxit,B,g0)

N = size(X,1);
% Minibatch iterates & errors
g = g0; G = cell(ceil(N/B),maxit+1); for n=1:ceil(N/B), G{n,1} = g; end
e = Y - linf(X,g); ev = Yv - linf(Xv,g);
E = NaN(2,ceil(N/B),maxit+1); E(:,end,1) = [e(:)'*e(:);ev(:)'*ev(:)]/2;
for i=1:maxit
  NN = randperm(N);					% Shuffle points
  n0 = 1;
  for n=1:ceil(N/B)
    idx = NN(n0:min(N,n0+B-1));				% Minibatch
    g.W = g.W + eta*e(idx,:)'*X(idx,:);
    g.w = g.w + eta*sum(e(idx,:),1)';			% SGD step
    e = Y - linf(X,g); ev = Yv - linf(Xv,g);
    G{n,i+1} = g; E(:,n,i+1) = [e(:)'*e(:);ev(:)'*ev(:)]/2;
    n0 = n0+B;
  end
end

