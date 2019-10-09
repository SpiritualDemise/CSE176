% X = lapeig(L,W[,nl]) Laplacian eigenmaps
%
% Laplacian eigenmaps algorithm of Belkin & Niyogi, Neural Comp. 2003.
%
% To create a Gaussian affinity matrix for a given high-dim dataset Y of NxD,
% the function "gaussaff" is useful. Examples:
%   W = gaussaff(Y,{'k',K},s) -> symmetric k-nn Gaussian affinities, width s
%   W = gaussaff(Y,{'k',K},s,a) -> ditto but with diffusion-map normalisation
%   W = gaussaff(Y,{'k',Inf},s) -> full-graph Gaussian affinities, width s
%   W = gaussaff(Y,{'k',K},Inf) -> symmetric k-nn binary affinities.
%
% In:
%   L: dimension of the latent space.
%   W: affinity matrix of NxN.
%   nl: normalised (1) or unnormalised (0) Laplacian. Default: 1.
% Out:
%   X: NxL matrix containing N D-dim data points rowwise.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2010 by Miguel A. Carreira-Perpinan

function X = lapeig(L,W,nl)

% ---------- Argument defaults ----------
if ~exist('nl','var') | isempty(nl) nl = 1; end
% ---------- End of "argument defaults" ----------

D = sum(W,2); LL = diag(sparse(D)) - W;
if nl DD = diag(sparse(D.^(-1/2))); LL = DD*LL*DD; end
LL = (LL+LL')/2;		% Ensure symmetric

% Compute eigenvectors with eigs or eig (if LL is or not sparse, resp.) and
% remove the trailing (null) eigenvector.
if issparse(LL)
  opts.issym = 1; opts.isreal = 1; opts.disp = 0;
  [U,eval] = eigs(LL,L+1,'sa',opts); eval = diag(eval);	% 'sm' fails!?
else
  [U,eval] = eig(LL); eval = diag(eval);
end
[eval,ind] = sort(eval); eval = eval(2:L+1); U = U(:,ind(2:(L+1)));
if nl X = DD*U; else X = U; end

