% S = dmpsample(Q,A,P,K,T) Sample sequences from a discrete Markov process
%
% In:
%   Q,A,P: discrete Markov process parameters (see dmptrain.m).
%   K: number of sequences to generate.
%   T: length of each of the sequences.
% Out:
%   S: KxT array of chars, the sample sequences.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function S = dmpsample(Q,A,P,K,T)

S1 = zeros(K,T); Pc = cumsum(P); Ac = cumsum(A')';
SS = bsxfun(@le,rand(1,K),Pc);
for i=1:K S1(i,1) = find(SS(:,i),1); end
for j=2:T
  SS = bsxfun(@le,rand(1,K),Ac(S1(:,j-1),:)');
  for i=1:K S1(i,j) = find(SS(:,i),1); end
end
S = Q(S1);

