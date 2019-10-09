% [K, comp, compl] = econncomp(X,e)
% Connected components of a point set (Euclidean distance)
%
% Finds the connected components of an undirected graph (e-ball graph) where
% the points xn are the vertices and there is an edge between points xn and xm
% if the Euclidean distance d(xn,xm) < e.
%
% It uses a depth-first search and computes the edges (distances) on the fly.
% In the worst case (every point is farther than e from each other) it takes
% O(N²), in the best case (every point is closer than e from each other) it
% takes O(N).
%
% The result is the same as running conncomp(sqdist(X)<(e^2)) but more
% efficient, because not all N² distances have to be computed.
%
% In:
%   X: NxD matrix containing N D-dimensional data points rowwise.
%   e: a scalar, the threshold distance.
% Out:
%   K: number of connected components.
%   comp: Kx1 structure array containing the components, each element
%      (field "c") is a list of vertices.
%   compl: Nx1 list containing the component label for each vertex.

% Copyright (c) 2009 by Miguel A. Carreira-Perpinan

function [K, comp, compl] = econncomp(X,e)

N = size(X,1);
L = 1:N;
K = 0;
compl = zeros(N,1);

while ~isempty(L)
  M = L(1);				% Pick point from L
  L = L(2:end);				% L = setdiff(L,M);
  K = K + 1; C = M;
  while ~isempty(M)			% Compute Kth component
    if ~isempty(L)
      n = L(sqdist(X(M(1),:),X(L,:))<(e^2));	% Neighbours of M(1) in L
      M = union(n,M(2:end));			% M = setdiff(M,M(1)) U n;
      L = setdiff(L,n);
      C = union(C,n);
    else
      M = [];
    end;
  end
  comp(K).c = C;
  compl(C) = K;				% Vertex memberships
end

