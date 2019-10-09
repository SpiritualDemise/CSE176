% [G,E] = linfgd(X,Y,Xv,Yv,eta,maxit,g0)
% Train linear function y = f(x) = W.x+w with gradient descent
%
% In:
%   X: NxL matrix, N L-dim data points rowwise, training set inputs.
%   Y: NxD matrix, N D-dim data points rowwise, training set outputs.
%   Xv: MxL matrix, M L-dim data points rowwise, validation set inputs.
%   Yv: MxD matrix, M D-dim data points rowwise, validation set outputs.
%   eta: positive scalar, the learning rate (step size).
%   maxit: number of iterations to run.
%   g0: (struct) initial linear function (see definition in linf.m).
% Out:
%   G: 1x(maxit+1) cell array where G{i} is a struct containing the linear
%      function at iteration i-1 (see struct definition in linf.m).
%   E: 2x(maxit+1) array where E(1,i) and E(2,i) contain the training and
%      validation error at iteration i-1.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [G,E] = linfgd(X,Y,Xv,Yv,eta,maxit,g0)

g = g0; G = cell(1,maxit+1); G{1} = g;
e = Y - linf(X,g); ev = Yv - linf(Xv,g);
E = zeros(2,maxit+1); E(:,1) = [e(:)'*e(:);ev(:)'*ev(:)]/2;
for i=1:maxit
  g.W = g.W + eta*e'*X; g.w = g.w + eta*sum(e,1)';	% GD step
  e = Y - linf(X,g); ev = Yv - linf(Xv,g);
  G{i+1} = g; E(:,i+1) = [e(:)'*e(:);ev(:)'*ev(:)]/2;
end

