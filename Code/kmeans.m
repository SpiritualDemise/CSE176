% [C,L,e,code] = kmeans(X,K[,init,maxit,tol]) K-means clustering
%
% In:
%   X: NxD matrix containing N D-dimensional data points rowwise.
%   K: integer in [1,N] containing the desired number of clusters.
%   init: initialisation, one of the following (default: 'kmeans++'):
%      'kmeans++': Arthur & Vassilvitskii SODA 2007;
%      'rndlabels': L = random assignment;
%      'rndmeans': C = random means in the range of X;
%      'rndsubset': C = random subset of data points;
%      C: KxD matrix containing the initial K cluster means.
%      Note: kmeans(X,K,'kmeans++',0) provides the kmeans++ initialisation.
%   maxit: maximal number of iterations. Default: Inf.
%   tol: small positive number, tolerance in the relative change of the
%      error function to stop iterating. Default: 0.
% Out:
%   C: KxD matrix containing the K cluster means.
%   L: Nx1 list containing the cluster labels (1 to K). Thus,
%      L(n) = k says point Xn is in cluster k.
%   e: list of values (for each iteration) of the error function.
%      This is the sum over all clusters of the within-cluster sums of
%      point-to-mean distances.
%   code: stopping code: 0 (tolerance achieved), 1 (maxit exceeded).
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2017 by Miguel A. Carreira-Perpinan

function [C,L,e,code] = kmeans(X,K,init,maxit,tol)

% ---------- Argument defaults ----------
if ~exist('init','var') || isempty(init) init = 'kmeans++'; end;
if ~exist('maxit','var') || isempty(maxit) maxit = Inf; end;
if ~exist('tol','var') || isempty(tol) tol = 0; end;
% ---------- End of "argument defaults" ----------

% Initialisation
C = km_init(X,K,init); [L,e] = km_le(X,C);
if maxit == 0, code = 1; return; end

% Iteration
code = -1; i = 0;
while code < 0
  oldL = L; oldC = C;			% Keep if back up needed
  C = km_m(X,L,C);			% New means,...
  [L,e(end+1)] = km_le(X,C);		% ...labels and error value
  i = i + 1;
  % Stopping condition
  if e(end) >= (1-tol)*e(end-1), code = 0;
  elseif i >= maxit, code = 1;
  end
end

% Back up if the last iteration increased the error function. This can
% happen due to numerical inaccuracy if two points Xn are very close.
if e(end) > e(end-1), i = i - 1; e = e(1:end-1); C = oldC; L = oldL; end


% [L,e] = km_le(X,C) Labels and error function given means
%
% Input/output arguments as above.
% Out:
%   e: value of the error function.

function [L,e] = km_le(X,C)

if size(X,2)==1
  % Fast algorithm for 1D data using binary search to assign point X(n) to its
  % closest codebook entry (on the sorted codebook). It avoids constructing an
  % intermediate matrix of NxK and takes O(N.logK) runtime.
  % This can differ slightly from the general computation if there are ties,
  % which may be broken differently.
  [S,J] = sort(C);
  L = J(discretize(X,[-Inf;(S(2:end)+S(1:end-1))/2;Inf])); % needs Matlab2015a
  e = sum((X - C(L)).^2);
else
  % This takes O(N.K.D) runtime and creates an intermediate matrix of O(N.K)
  % containing the point-centroid distances.
  [e,L] = min(sqdist(X,C),[],2); e = sum(e);
end


% C = km_m(X,L,C) Means given labels
%
% Input/output arguments as above.

function C = km_m(X,L,C)

% $$$ % This takes runtime O(N.K+N.D) but is easier to vectorise.
% $$$ for k=1:size(C,1), C(k,:) = mean(X(L==k,:),1); end

% This takes runtime O(N.D) but is harder to vectorise.
% Yet, it is much faster with Matlab 2015b...
C = zeros(size(C)); c = zeros(size(C,1),1);
for n=1:size(X,1), C(L(n),:) = C(L(n),:) + X(n,:); c(L(n)) = c(L(n)) + 1; end
C = bsxfun(@rdivide,C,c);

% Dead means have no associated data points. Set each to a random data point.
z = find(isnan(C(:,1))); C(z,:) = X(randperm(size(X,1),length(z)),:);


% C = km_init(X,K,init) Initialise means
%
% Input/output arguments as above.

function C = km_init(X,K,init)

if ischar(init)
  [N,D] = size(X);
  switch init
   case 'kmeans++'	% C = subset of data points encouraging dispersion
    C = zeros(K,D); D2 = Inf(1,N); C(1,:) = X(randi(N),:);
    for i=2:K
      % update shortest distance for each point given new centroid
      dist = sqdist(C(i-1,:),X); D2(dist<D2) = dist(dist<D2);
      % sample new centroid propto D2
      cD2 = cumsum(D2); C(i,:) = X(sum(cD2(end)*rand>=cD2)+1,:);
    end
   case 'rndlabels'	% labels = random assignment
    C = km_m(X,[(1:K)';randi(K,N-K,1)],zeros(K,D));
   case 'rndmeans'	% C = uniform random points in the range of X
    m = min(X,[],1); M = max(X,[],1);
    C = bsxfun(@plus,m,bsxfun(@times,M-m,rand(K,D)));
   case 'rndsubset'	% C = random subset of data points
    % Works badly because often means compete inside clusters
    C = X(randperm(N,K),:);
  end
else
  C = init;		% C = user-provided
end

