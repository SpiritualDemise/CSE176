% [f,fX,e,sv,l,m,xi] = svm2train(X,Y,ker,C[,x0,dual]) Train binary SVM
%
% This trains an SVM for binary classification, linear or nonlinear (with the
% kernels listed below). For the linear SVM there are two cases:
% - Linearly separable case (C=Inf): optimal separating hyperplane, if one
%   exists (check the error code e to see whether the QP was feasible). It
%   doesn't return slack variables.
% - Not linearly separable case (0 < C < Inf): soft margin hyperplane.
%
% The SVM requires solving a quadratic program (QP). This uses quadprog and
% optimoptions from the Matlab Optimization Toolbox. I use Matlab's active-set
% algorithm for both the primal and dual because it is more reliable to find
% the active set accurately (i.e., the support vectors).
% -> Matlab has removed 'active-set', so I use 'interior-point-convex'.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   Y: Nx1 vector, N labels in {-1,+1}.
%   ker: cell array giving the kernel type and its hyperparameters, one of:
%      {'lin'}: linear SVM;
%      {'rbf',s}: RBF (Gaussian) kernel of width s > 0;
%      {'poly',q}: polynomial kernel of degree q >= 1.
%   C: nonnegative scalar, the penalty parameter. For linear SVMs, C=Inf means
%      the optimal separating hyperplane and C<Inf the soft margin hyperplane.
%   x0: initial value to solve the QP:
%      . primal: (L+1)x1 vector, the weight vector and bias (the bias is the
%        last element in x0).
%      . dual: Nx1 vector, the Lagrange multipliers for the constraints.
%      Default: up to Matlab.
%   dual: 0 to solve the primal (only 'lin'), 1 to solve the dual (any kernel).
%      Default: 0 for 'lin', 1 otherwise.
% Out:
%   f: (struct) the SVM, with fields:
%      type='svm2', ker='lin'/'rbf'/'poly', s or q (kernel hyperparameter),
%      . for 'lin': weight vector w (Lx1), bias w0 (scalar);
%      . for other kernels: weight vector w (Mx1), support vectors C (MxL),
%        where M is the number of support vectors.
%   fX: Nx1 vector, f(X).
%   e: quadprog's error code:
%      1 (KKT point), 0 (maxit exceeded), -2 (infeasible), -3 (unbounded), etc.
%   sv: list of indices in 1:N of the support vectors in X.
%   l: Lagrange multipliers.
%   m: margin=1/|f.w|, distance from the hyperplane to its closest point in X.
%   xi: Nx1 vector, slack variable (constraint violation) for each data point
%      (not returned for the optimal separating hyperplane).

% Copyright (c) 2015 by Miguel A. Carreira-Perpinan

function [f,fX,e,sv,l,m,xi] = svm2train(X,Y,ker,C,x0,dual)

% ---------- Argument defaults ----------
if ~exist('x0','var') x0 = []; end;
if ~exist('dual','var') || isempty(dual) dual = 0; end;
% ---------- End of "argument defaults" ----------

%options = optimoptions('quadprog','Algorithm','active-set',...
%                       'Display','off','MaxIter',1000);
options = optimoptions('quadprog','Algorithm','interior-point-convex',...
                       'Display','off','MaxIter',1000);

% Threshold to determine numerically which Lagrange multipliers are 0, and to
% make the kernel matrix positive semidefinite:
th = 1e-8;

f.type = 'svm2'; f.ker = ker{1}; [N,L] = size(X);

if ~strcmp(f.ker,'lin')

  % Gram matrix depending on the kernel type
  switch f.ker
   case 'rbf', f.s = ker{2}; K = exp(-sqdist(X)/(2*f.s*f.s));
   case 'poly', f.q = ker{2}; K = (X*X'+1).^f.q;
  end
  K = bsxfun(@times,Y,bsxfun(@times,K,Y')); K = (K+K')/2;
  
  % We solve the dual QP
  [l,~,e] = quadprog(K+eye(N)*th,-ones(N,1),...
                     [],[],...
                     Y',0,...
                     zeros(N,1),C*ones(N,1),x0,options);
  sv = find(l > th); f.C = X(sv,:);		% support vectors
  f.w = Y(sv).*l(sv);				% weight vector
  xi = max(0,1-K(:,sv)*l(sv));			% slacks
  
elseif ~dual					% linear SVM, primal QP
  
  if isinf(C)					% without slacks (separable)
    [Z,~,e,~,LL] = quadprog(...
      diag([ones(L,1);0]),zeros(L+1,1),...
      bsxfun(@times,[X ones(N,1)],-Y),-ones(N,1),...
      [],[],[],[],x0,options);
  else						% with slacks (not separable)
    [Z,~,e,~,LL] = quadprog(...
      diag([ones(L,1);zeros(N+1,1)]),[zeros(L+1,1);C*ones(N,1)],...
      [bsxfun(@times,[X ones(N,1)],-Y) -eye(N,N)],-ones(N,1),...
      [],[],[-Inf(L+1,1);zeros(N,1)],[],x0,options);
  end
  f.w = Z(1:L); f.w0 = Z(L+1);			% weight vector & bias
  if ~isinf(C) xi = Z(L+2:end); end		% slacks
  l = LL.ineqlin;				% Lagrange multipliers
  sv = find(l > th);				% support vectors

else						% linear SVM, dual QP
  YX = bsxfun(@times,X,Y);  K = YX*YX';		% Gram matrix
  [l,~,e] = quadprog(K+eye(N)*th,-ones(N,1),...
                     [],[],...
                     Y',0,...
                     zeros(N,1),C*ones(N,1),x0,options);
  sv = find(l > th);				% support vectors
  f.w = YX(sv,:)'*l(sv);			% weight vector
  sv1 = find((l>th)&(l<C-th)); f.w0 = mean(Y(sv1)-X(sv1,:)*f.w);	% bias
  if ~isinf(C) xi = max(0,1-bsxfun(@times,X,Y)*f.w-f.w0*Y); end	% slacks

end

m = 1/norm(f.w);				% margin

if nargout>1 fX = svm2(X,f); end

