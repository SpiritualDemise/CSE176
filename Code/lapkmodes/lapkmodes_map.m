% ZZ = lapkmodes_map(XX,C,Z,X,s,lambda,k)
% Laplacian K-modes clustering: out-of-sample mapping
%
% This assigns new points XX to the clusters previously found by running
% Laplacian K-modes on a dataset X. C and Z are the outputs of lapkmodes.m
% when ran using as inputs X, s, lambda and L (a graph Laplacian constructed
% using a k-nearest-neighbor graph).
%
% Notes:
% - This assumes that the graph Laplacian L used in lapkmodes.m was a
%   nearest-neighbor graph constructed using k nearest neighbors.
%
% In:
%   XX: MxD matrix containing M D-dimensional test points rowwise.
%   C: KxD matrix containing the K centroids found during training.
%   Z: NxK matrix containing the soft assignments of the N training points to 
%      the K clusters.
%   X: NxD matrix containing N D-dimensional training points rowwise.
%   s: positive scalar, the bandwidth for the kernel density estimate.
%   lambda: trade-off parameter of the Laplacian smoothing term.
%   k: number of nearest neighbors used to construct the graph Laplacian
%      (here, needed to construct the affinity between each test point and
%      the training set).
% Out:
%   ZZ: NxK matrix containing soft assignment of test points to the K
%      clusters.

% Copyright (c) 2015 by Weiran Wang and Miguel A. Carreira-Perpinan

function ZZ = lapkmodes_map(XX,C,Z,X,s,lambda,k)

% Compute the Laplacian term in the out-of-sample mapping
N = size(X,1); M = size(XX,1); DIS = sqdist(XX,X);  
[DIS,idx] = sort(DIS,2,'ascend'); idx = idx(:,1:k);
W1 = sparse(M,N); idx = sub2ind([M,N],repmat((1:M)',k,1),idx(:));
W1(idx) = exp(-DIS(:,1:k)/(2*s^2));		% Heat kernel weighting
rs1 = sum(W1,2); P = diag(sparse(1./rs1))*W1;

% Compute the KDE term in the out-of-sample mapping
W2 = exp(-sqdist(XX,C)/(2*s^2)); rs2 = sum(W2,2); Q = diag(sparse(1./rs2))*W2;

% Compute the weight for combining the two terms
gamma = (rs2./rs1)/(2*lambda);

% Compute the out-of-sample mapping
ZZ = SimplexProj(P*Z + diag(sparse(gamma))*Q);

