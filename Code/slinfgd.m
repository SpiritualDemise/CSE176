% [G,E,C] = slinfgd(X,Y,Xv,Yv,o,eta,maxit,g0)
% Train logistic regression for binary classification y = s(w'.x+w0)
% with gradient descent, by maximum likelihood (min. cross-entropy) or by
% least-squares errors.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise, training set inputs.
%   Y: Nx1 matrix, N labels in {0,1}, training set outputs.
%   Xv: MxL matrix, M L-dim data points rowwise, validation set inputs.
%   Yv: Mx1 matrix, M labels in {0,1}, validation set outputs.
%   o: objective function (0: maximum likelihood, 1: least-squares errors).
%   eta: positive scalar, the learning rate (step size).
%   maxit: number of iterations to run.
%   g0: (struct) initial logistic regressor (see definition in slinf.m).
% Out:
%   G: 1x(maxit+1) cell array where G{i} is a struct containing the logistic
%      regressor at iteration i-1 (see struct definition in slinf.m).
%   E: 2x(maxit+1) array where E(1,i) and E(2,i) contain the training and
%      validation error at iteration i-1.
%   C: 2x(maxit+1) array where C(1,i) and C(2,i) contain the training and
%      validation classification error (in [0,1]) at iteration i-1.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [G,E,C] = slinfgd(X,Y,Xv,Yv,o,eta,maxit,g0)

g = g0; G = cell(1,maxit+1); G{1} = g; N = length(Y); M = length(Yv);

if ~o								% max-llh
  T = slinf(X,g); e = -sum( Y.*log(T) + (1-Y).*log(1-T) );
  Tv = slinf(Xv,g); ev = -sum( Yv.*log(Tv) + (1-Yv).*log(1-Tv) );
  E = zeros(2,maxit+1); E(:,1) = [e;ev];
  C = zeros(2,maxit+1);
  C(1,1) = sum(Y~=(T>0.5))/N; C(2,1) = sum(Yv~=(Tv>0.5))/M;
  for i=1:maxit
    A = Y - T;
    g.W = g.W + eta*A'*X; g.w = g.w + eta*sum(A);		% GD step
    T = slinf(X,g); e = -sum( Y.*log(T) + (1-Y).*log(1-T) );
    Tv = slinf(Xv,g); ev = -sum( Yv.*log(Tv) + (1-Yv).*log(1-Tv) );
    G{i+1} = g; E(:,i+1) = [e;ev];
    C(1,i+1) = sum(Y~=(T>0.5))/N; C(2,i+1) = sum(Yv~=(Tv>0.5))/M;
  end
else								% lsq-err
  T = slinf(X,g); e = Y - T;
  Tv = slinf(Xv,g); ev = Yv - Tv;
  E = zeros(2,maxit+1); E(:,1) = [e'*e;ev'*ev]/2;
  C = zeros(2,maxit+1);
  C(1,1) = sum(Y~=(T>0.5))/N; C(2,1) = sum(Yv~=(Tv>0.5))/M;
  for i=1:maxit
    A = (Y-T).*T.*(1-T);
    g.W = g.W + eta*A'*X; g.w = g.w + eta*sum(A);		% GD step
    T = slinf(X,g); e = Y - T;
    Tv = slinf(Xv,g); ev = Yv - Tv;
    G{i+1} = g; E(:,i+1) = [e'*e;ev'*ev]/2;
    C(1,i+1) = sum(Y~=(T>0.5))/N; C(2,i+1) = sum(Yv~=(Tv>0.5))/M;
  end
end

