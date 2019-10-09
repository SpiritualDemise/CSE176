% [G,E] = mlpgd(X,Y,Xv,Yv,l,eta,maxit,g0,I)
% Train sigmoidal MLP y = f(x) = V.s(W.x+w)+v with gradient descent
%
% In:
%   X: NxL matrix, N L-dim data points rowwise, training set inputs.
%   Y: NxD matrix, N D-dim data points rowwise, training set outputs.
%   Xv: MxL matrix, M L-dim data points rowwise, validation set inputs.
%   Yv: MxD matrix, M D-dim data points rowwise, validation set outputs.
%   l: (nonnegative scalar) weight decay parameter.
%   eta: positive scalar, the learning rate (step size).
%   maxit: number of iterations to run.
%   g0: (struct) initial linear function (see definition in mlp.m).
%   I: integer > 0, save gradient & error every Ith iteration. Make it equal
%      to maxit to save only the last iteration.
% Out:
%   G: 1x(floor(maxit/I)+1) cell array where G{i} is a struct containing the
%      linear function at iteration i-1 (see struct definition in mlp.m).
%   E: 2x(floor(maxit/I)+1) array where E(1,i) and E(2,i) contain the training
%      and validation error at iteration i-1.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [G,E] = mlpgd(X,Y,Xv,Yv,l,eta,maxit,g0,I)

% The following single call gives the same result after training for maxit
% iterations as the code below, but it doesn't save the intermediate MLPs or
% the validation errors: [g,~,e] = mlptrain(X,Y,g0,l,eta,N,0,maxit).

N = size(X,1);
g = g0; G = cell(1,floor(maxit/I)+1); G{1} = g;
e = Y - mlp(X,g); ev = Yv - mlp(Xv,g);
E = zeros(2,floor(maxit/I)+1); E(:,1) = [e(:)'*e(:);ev(:)'*ev(:)]/2;
for i=2:length(G)
  [g,~,e] = mlptrain(X,Y,g,l,eta,N,0,I);		% GD step & train.err.
  ev = Yv - mlp(Xv,g);					% Validation error
  G{i} = g; E(:,i) = [e(end);ev(:)'*ev(:)]/2;
end

