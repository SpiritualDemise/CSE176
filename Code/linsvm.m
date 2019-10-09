% Y = linsvm(X,f) Value of binary linear SVM f(x) = w'.x + w0
%
% In:
%   X: NxD matrix, N D-dim data points rowwise.
%   f: (struct) the binary SVM, with fields:
%      type='linsvm', w (Dx1), w0 (scalar).
% Out:
%   Y: Nx1 matrix, N real-valued outputs Y = f(X).
%      To obtain the sign in {-1,+1} do Y = 2*(Y > 0) - 1.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function Y = linsvm(X,f)

Y = X*f.w + f.w0;

