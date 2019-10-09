% [e,C,Cn,L] = confumat(K,Y,L) or (K,Y,P) Confusion matrix & classif. error
%
% Y(n) contains the ground-truth label (in 1:K) for point X(n). The values of
% X are not necessary. The predicted label can be given in one of two ways:
% - As labels: L(n) is the label in 1:K.
% - As posterior probabilities: row n of matrix P contains the posterior
%   probability p(k|X(n)), for each class k=1:K, for vector X(n). Hence,
%   each row contains K values in [0,1] that sum to 1. The predicted label
%   has the largest p(k|X(n)).
%
% In:
%   K: number of classes (assumed 1:K), K >= 2.
%   Y: Nx1 list of N ground-truth labels (in 1:K).
%   L: Nx1 list of N predicted labels (in 1:K).
%   P: NxK matrix of posterior probabilities. For K=2, you may just give
%      P(:,1) as argument.
% Out:
%   e: classification error in [0,1].
%   C: KxK confusion matrix, unnormalised (raw counts).
%   Cn: KxK confusion matrix, normalised (each row in [0,1] with sum 1).
%   L: Nx1 list of N predicted labels (in 1:K).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [e,C,Cn,L] = confumat(K,Y,P)

N = length(Y);
% Get predicted labels
if size(P,2)==1
  if (K>2) | any(P>1), L = P;			% L
  else L = (P<0.5) + 1;				% L or P(C=1)
  end
else [~,L] = max(P,[],2);			% P
end

C = zeros(K,K); for n=1:N, C(Y(n),L(n)) = C(Y(n),L(n)) + 1; end
%for k=1:K, C(k,:) = histc(L(Y==k),1:K); end;	% slower

e = 1 - sum(diag(C))/N;

if nargout>2, Cn = bsxfun(@rdivide,C,sum(C,2)); end

