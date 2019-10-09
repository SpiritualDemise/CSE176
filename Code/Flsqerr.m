% function [f,g,H] = Flsqerr(X,h,T,Y) Least-squares error function
%
% f(x) = ½sum^N_{n=1}{ |yn - h(tn;x)|² }
% where x are the parameters of the model h, and t and y are its input and
% output arguments.
% Currently this works when h is 'linf' or 'slinf'.
%
% In:
%   X: N x n list of row vectors.
%   h: function (see format in linftrain.m).
%   T: M x D list of row vectors (inputs to h).
%   Y: M x E list of row vectors (outputs to h).
% Out:
%   f: N x 1 list of function values.
%   g: N x n list of gradient vectors.
%   H: n x n x N list of Hessian matrices.

% Copyright (c) 2016 by Miguel A. Carreira-Perpinan

function [f,g,H] = Flsqerr(X,h,T,Y)

N = size(X,1); D = size(T,2); E = size(Y,2);
f = zeros(N,1);
fcn = str2func(h.type);		% Use string as function handle
for n=1:N
  % Form the model struct, this part works for 'linf' and 'slinf'
  h.W = reshape(X(n,1:E*D),[E D]); h.w = X(n,end-E+1:end)'; 
  % Apply model to inputs to get outputs, this part works in general
  hY = fcn(T,h);
  % Least-squares error value
  f(n) = sum(sum((Y - hY).^2))/2;
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
