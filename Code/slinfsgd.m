% [G,E,C] = slinfsgd(X,Y,Xv,Yv,o,eta,maxit,B,g0)
% Train logistic regression for binary classification y = s(w'.x+w0)
% with stochastic gradient descent, by maximum likelihood (min. cross-entropy)
% or by least-squares errors.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise, training set inputs.
%   Y: Nx1 matrix, N labels in {0,1}, training set outputs.
%   Xv: MxL matrix, M L-dim data points rowwise, validation set inputs.
%   Yv: Mx1 matrix, M labels in {0,1}, validation set outputs.
%   o: objective function (0: maximum likelihood, 1: least-squares errors).
%   eta: positive scalar, the learning rate (step size).
%   maxit: number of iterations to run.
%   B: minibatch size, between 1 and N.
%   g0: (struct) initial logistic regressor (see definition in slinf.m).
% Out:
%   G: (ceil(N/B))x(maxit+1) cell array where G{n,i} is a struct containing
%      the logistic regressor at minibatch n and iteration i-1 (see struct
%      definition in slinf.m).
%   E: 2x(ceil(N/B))x(maxit+1) array where E(1,n,i) and E(2,n,i) contain the
%      training and validation error at minibatch n and iteration i-1.
%   C: 2x(ceil(N/B))x(maxit+1) array where C(1,n,i) and C(2,n,i) contain the
%      training and validation classification error (in [0,1]) at minibatch
%      n and iteration i-1.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [G,E,C] = slinfsgd(X,Y,Xv,Yv,o,eta,maxit,B,g0)

N = size(X,1); N = length(Y); M = length(Yv);
% Minibatch iterates & errors
g = g0; G = cell(ceil(N/B),maxit+1); for n=1:ceil(N/B), G{n,1} = g; end

if ~o								% max-llh
  T = slinf(X,g); e = -sum( Y.*log(T) + (1-Y).*log(1-T) );
  Tv = slinf(Xv,g); ev = -sum( Yv.*log(Tv) + (1-Yv).*log(1-Tv) );
  E = NaN(2,ceil(N/B),maxit+1); E(:,end,1) = [e;ev];
  C = NaN(2,ceil(N/B),maxit+1);
  C(1,end,1) = sum(Y~=(T>0.5))/N; C(2,end,1) = sum(Yv~=(Tv>0.5))/M;
  for i=1:maxit
    NN = randperm(N);					% Shuffle points
    n0 = 1;
    for n=1:ceil(N/B)
      idx = NN(n0:min(N,n0+B-1));				% Minibatch
      A = Y(idx) - T(idx);
      g.W = g.W + eta*A'*X(idx,:); g.w = g.w + eta*sum(A);	% SGD step
      T = slinf(X,g); e = -sum( Y.*log(T) + (1-Y).*log(1-T) );
      Tv = slinf(Xv,g); ev = -sum( Yv.*log(Tv) + (1-Yv).*log(1-Tv) );
      G{n,i+1} = g; E(:,n,i+1) = [e;ev];
      C(1,n,i+1) = sum(Y~=(T>0.5))/N; C(2,n,i+1) = sum(Yv~=(Tv>0.5))/M;
      n0 = n0+B;
    end
  end
else								% lsq-err
  T = slinf(X,g); e = Y - T;
  Tv = slinf(Xv,g); ev = Yv - Tv;
  E = NaN(2,ceil(N/B),maxit+1); E(:,end,1) = [e'*e;ev'*ev]/2;
  C = NaN(2,ceil(N/B),maxit+1);
  C(1,end,1) = sum(Y~=(T>0.5))/N; C(2,end,1) = sum(Yv~=(Tv>0.5))/M;
  for i=1:maxit
    NN = randperm(N);					% Shuffle points
    n0 = 1;
    for n=1:ceil(N/B)
      idx = NN(n0:min(N,n0+B-1));				% Minibatch
      A = (Y(idx)-T(idx)).*T(idx).*(1-T(idx));
      g.W = g.W + eta*A'*X(idx,:); g.w = g.w + eta*sum(A);	% SGD step
      T = slinf(X,g); e = Y - T;
      Tv = slinf(Xv,g); ev = Yv - Tv;
      G{n,i+1} = g; E(:,n,i+1) = [e'*e;ev'*ev]/2;
      C(1,n,i+1) = sum(Y~=(T>0.5))/N; C(2,n,i+1) = sum(Yv~=(Tv>0.5))/M;
      n0 = n0+B;
    end
  end
end

