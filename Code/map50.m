% M = map50(N) 50% saturation colormap
%
% Creates a colormap with N muted colours (spanning the entire hue range,
% with 50% saturation and value 1). Useful for 2D plots of regions (eg from
% a clustering or classification algorithm, or from a political map). The
% region colours are unobtrusive, so annotations are visible.
%
% In:
%   N: number of colours to create.
% Out:
%   M: Nx3 colormap.

% Copyright (c) 2015 by Miguel A. Carreira-Perpinan

function M = map50(N)

M = hsv2rgb([((1:N)'-1)/N ones(N,1)*0.5 ones(N,1)]);

