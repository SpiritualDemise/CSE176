% L = knn(X,Y,x[,K,p]) k-nearest neighbour classifier
%
% In:
%   X: NxD matrix containing N D-dimensional data points rowwise (training).
%   Y: Nx1 matrix containing the class labels for X (in 1..K).
%   x: MxD matrix containing M D-dimensional data points rowwise (test).
%   K: (scalar) number of nearest neighbours to use. Default: 1.
%   p: (nonnegative scalar) use Lp distance. Default: 2.
% Out:
%   L: Mx1 matrix containing the predicted class labels for x.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function L = knn(X,Y,x,K,p)

% ---------- Argument defaults ----------
if ~exist('K','var') || isempty(K) K = 1; end;
if ~exist('p','var') || isempty(p) p = 2; end;
% ---------- End of "argument defaults" ----------

M = size(x,1); L = zeros(M,1); N = size(X,1);

% Block size (see below), select as large as your memory allows
mem = 1; B = floor((mem*1024^3)/(4*N*8)/2);	% This will fit in mem GB RAM

if p==2		% fast with distance vectorisation + loop over blocks of x
  i1 = 1; i2 = min(M,B); X2s = sum(X.^2,2)';
  while i1 <= M
    if K==1	% nearest-neighbour classifier, min faster than sort
      [~,I] = min(bsxfun(@minus,X2s,2*x(i1:i2,:)*X'),[],2);
      L(i1:i2) = Y(I);
    else
      [~,I] = sort(bsxfun(@minus,X2s,2*x(i1:i2,:)*X'),2);
      L(i1:i2) = mode(Y(I(:,1:K)),2);
    end
    i1 = i1 + B; i2 = min(M,i1+B-1);
  end
else		% loop over rows of x, not vectorised
  if K==1
    for m=1:M
      [~,I] = min(sum(abs(bsxfun(@minus,x(m,:),X)).^p,2));
      L(m) = Y(I);
    end
  else
    for m=1:M
      [~,I] = sort(sum(abs(bsxfun(@minus,x(m,:),X)).^p,2));	% K-nn
      L(m) = mode(Y(I(1:K)));					% max votes
    end
  end
end

