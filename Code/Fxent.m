% function [f,g,H] = Fxent(X,h,T,Y) Cross-entropy error function
%
% f(x) = - sum^N_{n=1}{ yn.log(h(tn;x)) + (1-yn).log(1-h(tn;x)) }
% where x are the parameters of the model h, and t and y are its input and
% output arguments.
% Currently this works when h is 'slinf'.
%
% In:
%   X: N x n list of row vectors.
%   h: function (see format in slinftrain.m).
%   T: M x D list of row vectors (inputs to h).
%   Y: M x 1 list of row vectors (outputs to h).
% Out:
%   f: N x 1 list of function values.
%   g: N x n list of gradient vectors.
%   H: n x n x N list of Hessian matrices.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [f,g,H] = Fxent(X,h,T,Y)

N = size(X,1);
f = zeros(N,1);
fcn = str2func(h.type);		% Use string as function handle
for n=1:N
  % Form the model struct, this part works for 'slinf'
  h.W = X(n,1:end-1); h.w = X(n,end);
  % Apply model to inputs to get outputs, this part works in general
  hY = fcn(T,h);
  % Cross-entropy value
  f(n) = -sum( Y.*log(hY) + (1-Y).*log(1-hY) );
end

% $$$ % Finish this
% $$$ if nargout > 1
% $$$   if h.type == 'linf'
% $$$   g = bsxfun(@plus,X*A',b');
% $$$   if nargout > 2
% $$$     H = repmat(A,[1 1 N]);
% $$$   end
% $$$ end
% $$$ 
