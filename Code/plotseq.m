% function h = plotseq(X[,s]) Plot a sequence of points in 2D
%
% This plots the sequence on top of the current figure; it is useful to
% overlay the sequence on a contour plot.
%
% In:
%   X: Nx2 list of row vectors.
%   s: string containing the line style. Default: 'ro-'.
% Out:
%   h: handle for the plot.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2005 by Miguel A. Carreira-Perpinan

function h = plotseq(X,s)

% ---------- Argument defaults ----------
if ~exist('s','var') | isempty(s) s = 'ro-'; end;
% ---------- End of "argument defaults" ----------

hold on; h = plot(X(:,1),X(:,2),s); hold off;

