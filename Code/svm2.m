% [Y,J] = svm2(X,f[,B]) Value and Jacobian wrt x of binary SVM f(x)
%
% This handles both linear and kernel SVMs for binary classification.
% See svm2train.
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   f: (struct) the binary SVM.
%   B: 0, return real labels in (-Inf,Inf); 1, return their sign in {-1,+1}.
%      Default: 1.
% Out:
%   Y: Nx1 matrix, N outputs Y = f(X) in {-1,+1}.
%   J: DxL Jacobian matrix for the real-valued labels (assumes N=1 input only).

% Copyright (c) 2015 by Miguel A. Carreira-Perpinan

function [Y,J] = svm2(X,f,B)

% ---------- Argument defaults ----------
if ~exist('B','var') || isempty(B) B = 1; end;
% ---------- End of "argument defaults" ----------

switch f.ker
 case 'lin', Y = X*f.w + f.w0;
 case 'poly', Y = ((X*f.C'+1).^f.q)*f.w;
 case 'rbf', Y = exp(-sqdist(X,f.C)/(2*f.s*f.s))*f.w;
end
if B, Y = 2*(Y > 0) - 1; end

if nargout>1 J = f.w; end	% TODO for nonlinear kernel

