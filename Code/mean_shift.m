% [C,L] = mean_shift(X,s) Mean-shift clustering
%
% In:
%   X: NxD matrix containing N D-dimensional data points rowwise.
%   s: bandwidth.
% Out:
%   C: KxD matrix containing the K cluster modes.
%   L: Nx1 list containing the cluster labels (1 to K). Thus,
%      L(n) = k says point Xn is in cluster k.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [C,L] = mean_shift(X,s)

% Estimate a kernel density estimate (KDE) with bandwidth s
N = size(X,1); kde.p = ones(N,1)/N; kde.c = X; kde.S = s^2; kde.type = 'i';
[C,~,~,~,~,L] = GMmodes(kde);	% find modes of the KDE

