% [s,ds] = sigmoid(x) Sigmoid value and derivative.
%
% In:
%   x: any array.
% Out:
%   s, ds: (same shape as x) value and derivative of the sigmoid at each x.

% Copyright (c) 2008 by Miguel A. Carreira-Perpinan

function [s,ds] = sigmoid(x)

s = (1+exp(-x)).^(-1); ds = s.*(1-s);

