% [Q,A,P] = dmptrain(X) Train a discrete Markov process
%
% In:
%   X: 1xC cell array of strings (not necessarily of the same length).
% Out:
%   Q: 1xN array of chars, the states.
%   A: NxN matrix, the transition probabilities between the N states.
%   P: Nx1 vector, the probabilities of the initial state distribution.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [Q,A,P] = dmptrain(X)

% States
Q = unique(cell2mat(X)); N = length(Q);

% Learn initial state distribution
P = zeros(N,1);
for i=1:length(X) P(Q==X{i}(1)) = P(Q==X{i}(1)) + 1; end
P = P/sum(P);

% Learn transition matrix
A = zeros(N,N);
for i=1:length(X)
  if length(X{i}) > 1
    [I1,I2] = find(bsxfun(@eq,X{i}',Q)); I = sortrows([I1 I2],1); I = I(:,2);
    for j=sub2ind([N N],I(1:end-1),I(2:end))' A(j) = A(j) + 1; end
  end
end
A(isnan(A)) = 1; A = bsxfun(@rdivide,A,sum(A,2));

