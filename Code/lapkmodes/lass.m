% [Z,Y,U,nu] = lass(L,l,G[,r,Y,U,dochol,maxit,tol])
% Laplacian assignment problem
%
% Solves a QP over soft assignments Z:
%   min_Z { l.tr(Z'.L.Z) - tr(G'.Z) } st Z.1=1, Z>=0
% where:
% - Z of NxK is the assignment matrix (znk\in[0,1] is the soft assignment of
%   item n to object k).
% - L of NxN is a graph Laplacian matrix, encouraging nearby items to have
%   similar assignments.
% - l >= 0 is the regularisation weight for the Laplacian term.
% - G of NxK is a data matrix (gnk = similarity of item n to object k, can be
%   positive, negative or zero).
%
% Upon convergence, Z is an optimal solution and its Lagrange multipliers are
% nu (for the equality constraints) and r*U (for the inequality constraints),
% under the convention that the multipliers appear with a plus sign in the
% Lagrangian.
%
% The algorithm uses the alternating direction method of multipliers (ADMM),
% with a direct solver (using Schur's complement) and caching the (permuted)
% Cholesky factor of an NxN matrix related to the graph Laplacian L.
%
% Notes:
% - This uses a default penalty parameter r=1, but a better value is given
%   in the paper.
% - If the graph associated with L is disconnected, it would be better to
%   run LASS on each connected component.
% - The solution Z returned may be inaccurate and infeasible if tol is low.
%   If you want to obtain a feasible point, you can project it onto the
%   simplex as follows: Z = SimplexProj(Z);
% - If this QP is embedded in an outer loop (as happens in the Laplacian
%   k-modes clustering algorithm), one should use as initial values for
%   Z,Y,U,nu the result from the previous QP's iterate (warm start).
%   (Actually, all the algorithm needs are the initial Y and U.)
%
% In:
%   L: NxN posdef matrix, (usually sparse) graph Laplacian.
%   l: nonnegative scalar, Laplacian regularisation weight.
%   G: NxK matrix, item-object similarities.
%   r: positive penalty parameter. Default: 1.
%   Y,U: NxK matrices, initial values for Y,U (see output arguments).
%      Default: zeros.
%   dochol: 1 to use the permuted Cholesky factor, 0 to use a direct solver.
%      Default: 1.
%   maxit: maximal number of iterations. Default: 1000.
%   tol: small positive number, tolerance in the change of Z to stop iterating.
%      Default: 1e-5.
% Out:
%   Z: NxK matrix, item-object assignments.
%   Y: NxK matrix, auxiliary variables for projection on nonnegative orthant.
%   U: NxK matrix, multipliers for Y.
%   nu: Nx1 vector, Lagrange multipliers for the equality constraints.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2014 by Miguel A. Carreira-Perpinan

function [Z,Y,U,nu] = lass(L,l,G,r,Y,U,dochol,maxit,tol)

[N,K] = size(G);
if l==0						% exact solution, much faster
  [nu,I] = max(G,[],2); Y = zeros(N,K); Y(sub2ind([N K],1:N,I')) = 1;
  U = bsxfun(@minus,G,nu)/r; Z = Y; return;
end
% ---------- Argument defaults ----------
if ~exist('r','var') || isempty(r) r = 1; end;
if ~exist('Y','var') || isempty(Y) Y = zeros(N,K); U = Y; end;
if ~exist('dochol','var') || isempty(dochol) dochol = 1; end;
if ~exist('maxit','var') || isempty(maxit) maxit = 1000; end;
if ~exist('tol','var') || isempty(tol) tol = 1e-5; end;
% ---------- End of "argument defaults" ----------

LI = 2*l*L+r*speye(N,N); h = (-sum(G,2)+r)/K; Zold = zeros(N,K);
if dochol
  if issparse(LI)
    [R,~,S] = chol(LI); Rt = R';		% do Cholesky factorization
    h = S'*h; G = S'*G; Y = S'*Y; U = S'*U;	% and permute variables
    for i=1:maxit
      nu = (r/K)*sum(Y-U,2) - h;
      Z = R \ (Rt \ bsxfun(@minus,r*(Y-U)+G,nu)); % triangular backsolves
      Y = max(Z+U,0);
      U = U + Z - Y;
      if max(abs(Z(:)-Zold(:))) < tol break; end; Zold = Z;
    end
    Z = S*Z; Y = S*Y; U = S*U; nu = S*nu;	% undo permutation
  else						% same, without preordering
    R = chol(LI); Rt = R';
    for i=1:maxit
      nu = (r/K)*sum(Y-U,2) - h;
      Z = R \ (Rt \ bsxfun(@minus,r*(Y-U)+G,nu));
      Y = max(Z+U,0);
      U = U + Z - Y;
      if max(abs(Z(:)-Zold(:))) < tol break; end; Zold = Z;
    end
  end
else
  for i=1:maxit
    nu = (r/K)*sum(Y-U,2) - h;
    Z = LI \ bsxfun(@minus,r*(Y-U)+G,nu);	% direct solution
    Y = max(Z+U,0);
    U = U + Z - Y;
    if max(abs(Z(:)-Zold(:))) < tol break; end; Zold = Z;
  end
end

%% The solution Z may be slightly inaccurate and infeasible, particularly if
%% tol was low. If you want to obtain a feasible point, you can project it
%% onto the simplex and compute the Lagrange multipliers (under the convention
%% that they appear with a plus sign in the Lagrangian) as follows:
Z = SimplexProj(Z);
D = (Z.^2 + 1).^(-1); Q = bsxfun(@plus,2*l*L*Z-G,sum(G,2)/K);
M = D.*Q - bsxfun(@times,D,(sum(Q,2)-sum(D.*Q,2))./(K-sum(D,2)));
nu = -(sum(M,2)+sum(G,2))/K;

