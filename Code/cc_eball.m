% [K,L] = cc_eball(X,e) Connected-components clustering (e-ball)
%
% In:
%   X: NxD matrix containing N D-dimensional data points rowwise.
%   e: minimum distance to connect two points.
% Out:
%   K: number of clusters.
%   L: Nx1 list containing the cluster labels (1 to K). Thus,
%      L(n) = k says point Xn is in cluster k.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [K,L] = cc_eball(X,e)

[K,~,L] = econncomp(X,e);	% conncomp(sqdist(X)<(e4^2));

