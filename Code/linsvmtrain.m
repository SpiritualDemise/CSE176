% [f,e,sv,l,m,xi] = linsvmtrain(X,Y,C,dual) Train binary linear SVM
%
% This trains a linear SVM for binary classification. There are two cases:
% - Linearly separable case (C=NaN): optimal separating hyperplane, if one
%   exists (check the error code e to see whether the QP was feasible). It
%   doesn't return slack variables.
% - Not linearly separating case (0 < C < Inf): soft margin hyperplane.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   Y: Nx1 vector, N labels in {-1,+1}.
%   C: nonnegative scalar, penalty parameter for the soft margin hyperplane.
%      C = NaN means the optimal separating hyperplane.
%   dual: 0 to solve the primal, 1 to solve the dual.
% Out:
%   f: (struct) the SVM.
%   e: quadprog's error code:
%      1 (KKT point), 0 (maxit exceeded), -2 (infeasible), -3 (unbounded), etc.
%   sv: list of indices in 1:N of the support vectors in X.
%   l: Nx1 vector, the Lagrange multiplier for each data point.
%   m: margin=1/|f.w|, distance from the hyperplane to its closest point in X.
%   xi: Nx1 vector, slack variable (constraint violation) for each data point
%      (not returned for the optimal separating hyperplane).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [f,e,sv,l,m,xi] = linsvmtrain(X,Y,C,dual)

% active-set has been removed from quadprog from Matlab version 2016a onwards
%options = optimoptions('quadprog','Algorithm','active-set','Display','off',...
%                       'MaxIter',1000);
options = optimoptions('quadprog','Display','off');

% Threshold to determine numerically which Lagrange multipliers are 0, and to
% make the kernel matrix positive semidefinite:
th = 1e-5;

f.type = 'linsvm'; [N,L] = size(X);

if ~dual					% linear SVM, primal QP
  
  if isnan(C)					% without slacks (separable)
    [Z,~,e,~,LL] = quadprog(...
      diag([ones(L,1);0]),zeros(L+1,1),...
      bsxfun(@times,[X ones(N,1)],-Y),-ones(N,1),...
      [],[],[],[],[],options);
  else						% with slacks (not separable)
    [Z,~,e,~,LL] = quadprog(...
      diag([ones(L,1);zeros(N+1,1)]),[zeros(L+1,1);C*ones(N,1)],...
      [bsxfun(@times,[X ones(N,1)],-Y) -eye(N,N)],-ones(N,1),...
      [],[],[-Inf(L+1,1);zeros(N,1)],[],[],options);
  end
  f.w = Z(1:L); f.w0 = Z(L+1);			% weight vector & bias
  if ~isnan(C) xi = Z(L+2:end); end		% slacks
  l = LL.ineqlin;				% Lagrange multipliers
  sv = find(l > th);				% support vectors

else						% linear SVM, dual QP

  YX = bsxfun(@times,X,Y);  K = YX*YX';		% Gram matrix
  if isnan(C)
    [l,~,e] = quadprog(K+eye(N)*th,-ones(N,1),...
                       [],[],...
                       Y',0,...
                       zeros(N,1),[],[],options);
  else
    [l,~,e] = quadprog(K+eye(N)*th,-ones(N,1),...
                       [],[],...
                       Y',0,...
                       zeros(N,1),C*ones(N,1),[],options);
  end
  sv = find(l > th);				% support vectors
  f.w = YX(sv,:)'*l(sv);			% weight vector
  sv1 = find((l>th)&(l<C-th)); f.w0 = mean(Y(sv1)-X(sv1,:)*f.w);% bias
  if ~isnan(C) xi = max(0,1-bsxfun(@times,X,Y)*f.w-f.w0*Y); end	% slacks

end

m = 1/norm(f.w);				% margin

