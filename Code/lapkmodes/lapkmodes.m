% [C,Z,E,C0,Z0] = lapkmodes(X,K,s,lambda,L[,tol,maxit,C0,Z0,method])
% Laplacian K-modes clustering of dataset X
%
% In:
%   X: NxD matrix containing N D-dimensional data points rowwise.
%   K: integer in [1,N] containing the desired number of clusters.
%   s: positive scalar, the bandwidth for the kernel density estimate.
%   lambda: trade-off parameter of the Laplacian smoothing term.
%   L: NxN matrix, the graph Laplacian built on the data set.
%   tol: small positive number, tolerance in the relative change of the
%      centroids to stop iterating. Default: 1e-3.
%   maxit: maximum number of iterations. Default: 100.
%   C0: KxD matrix containing the initial K centroids. Default: from k-means.
%   Z0: NxK matrix containing initial assignment values. Default: from C0.
%   method: method for solving the Z-step, 0 for gradient projection, 1 for
%     ADMM. Default: 0.
% Out:
%   C: KxD matrix containing final cluster centroids.
%   Z: NxK matrix containing final assignment values.
%   E: objective function values at each iteration.
%   C0, Z0: initial values for C and Z.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2015 by Weiran Wang and Miguel A. Carreira-Perpinan

function [C,Z,E,C0,Z0] = lapkmodes(X,K,s,lambda,L,tol,maxit,C0,Z0,method)

N = size(X,1);

% ---------- Argument defaults ----------
if ~exist('tol','var') || isempty(tol) tol = 1e-3; end;
if ~exist('maxit','var') || isempty(maxit) maxit = 100; end
if ~exist('C0','var') || isempty(C0)
  [C0,label] = kmeans(X,K);
  Z0 = zeros(N,K); Z0(sub2ind([N,K],(1:N)',label)) = 1;
end
if ~exist('Z0','var') || isempty(Z0)		% Voronoi partition given C0
  [~,label] = sort(sqdist(X,C0),2);
  Z0 = zeros(N,K); Z0(sub2ind([N,K],(1:N)',label)) = 1;
end
if ~exist('method','var') || isempty(method) method = 0; end
% ---------- End of "argument defaults" ----------

C = C0; Z = Z0;
WC = exp(-sqdist(X,C)/(2*s^2));		% affinity matrix between X and C
E = lambda*trace(Z'*L*Z) - trace(Z'*WC);

if lambda>0
  if method==0				% stepsize for gradient projection
    stepsize = 1/(2*lambda*max(eigs(L)));
  else					% penalty parameter for ADMM
    lm = eigs(L); sm = eigs(L,10,'SM'); sm = sm(sm>1e-8);
    rho = 2*lambda*sqrt(max(lm)*min(sm)); Ylass = Z; Ulass = zeros(N,K);
  end
end

for i=1:maxit
  
  oldC = C;
  
  % Optimize over C given Z: mean-shift
  for k=1:K
    zk = Z(:,k); zk = zk/sum(zk);		% Gaussian mixture weights
    C(k,:) = meanshift(X,s,zk,C(k,:),1e-5,1e3);
  end
  
  % Optimize over Z given C: quadratic program
  WC = exp(-sqdist(X,C)/(2*s^2));
  if lambda>0
    if method==0				% gradient projection
      Z = gradproj(Z,stepsize,lambda,L,WC,1e-6,1e3);
    else  					% ADMM
      [Z,Ylass,Ulass] = lass(L,lambda,WC,rho,Ylass,Ulass);
    end
  else						% lambda=0, becomes K-modes
    [~,label] = max(WC,[],2);
    Z = zeros(N,K); Z(sub2ind([N,K],(1:N)',label)) = 1;
  end

  E = [E,lambda*trace(Z'*L*Z)-trace(Z'*WC)];
  
  if norm(C-oldC,'fro') < tol*norm(oldC,'fro') break; end
  
end


% [c1,i] = meanshift(X,s,m,c0,tol,maxit)
% Apply mean-shift steps for mode finding

function c1 = meanshift(X,s,m,c0,tol,maxit)

for i=1:maxit
  W = m.*exp(-sqdist(X,c0)/(2*s^2));
  c1 = W'*X/sum(W);
  if max(abs(c1-c0)) < tol*max(abs(c0))
    break;
  else
    c0 = c1;
  end
end


% [Z,E] = gradproj(Z,stepsize,lambda,L,WC,tol,maxit)
% Apply gradient projection minimization

function [Z,E] = gradproj(Z,stepsize,lambda,L,WC,tol,maxit)

% ---------- Argument defaults ----------
if ~exist('tol','var') || isempty(tol) tol = 1e-4; end
if ~exist('maxit','var') || isempty(maxit) maxit = 1000; end
% ---------- End of "argument defaults" ----------

E = lambda*trace(Z'*L*Z) - trace(Z'*WC);

Y = Z; t = 1;					% Y is the auxiliary sequence

for i=1:maxit
  Zold = Z;
  grad = 2*lambda*L*Y-WC; Y = Y - stepsize*grad; Z = SimplexProj(Y);
  E = [E lambda*trace(Z'*L*Z)-trace(Z'*WC)];
  if abs(E(end-1)-E(end))<tol*abs(E(end-1)) break; end
  told = t; t = (1+sqrt(1+4*t^2))/2; Y = Z + (told-1)/t*(Z-Zold);
end

