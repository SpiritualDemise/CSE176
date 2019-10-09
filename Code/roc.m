% [C,T,A] = roc(Y,P) ROC curve
%
% Curve of values (fp,tp) for a classifier (false and true positive rates).
% Note that for a finite dataset (i.e., finite N) the "curve" only exists at
% at most N points in the plane.
%
% In:
%   Y: Nx1 list of N ground-truth labels (in 1:2).
%   P: Nx1 matrix of posterior probabilities for class 1, p(C=1|X(n)) in [0,1].
% Out:
%   C: Nx2 array containing the curve as (fp,tp) pairs.
%   T: Nx1 list of threshold values in [0,1] at which the curve is defined.
%   A: scalar, area under the curve (AUC).

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [C,T,A] = roc(Y,P)

N = length(Y); p = sum(Y==1); TP = p; n = N - p; FP = n; j = 1;
[P,I] = sort(P); Y = Y(I); T = unique([0;P;1]); tp = T; fp = tp; P = [P;Inf];
for i=1:length(T)
  while P(j) <= T(i)
    if Y(j)==1, TP = TP - 1; else FP = FP - 1; end
    j = j + 1;
  end
  tp(i) = TP; fp(i) = FP;
end
C = [fp/n tp/p];

if nargout>2, A = diff(C(end:-1:1,1))'*(C(end:-1:2,2)+C(end-1:-1:1,2))/2; end

