% AC(A1,B1,A2,B2,A3,B3,...) Compare pairs of arrays numerically (Inf-norm)
%
% It returns max(abs(Ai(:)-Bi(:))) for i = 1, 2... This is useful to verify
% that two different computations are essentially the same numerically, for
% example:
%   A = randn(10,10); B = A*inv(A)*A; AC(A,B)
% The argument list must contain an even number of arrays and Ai and Bi must
% have the same dimension for i = 1, 2...

% Copyright (c) 2017 by Miguel A. Carreira-Perpinan

function AC(varargin)

if rem(nargin,2)~=0 disp('need even number of arguments'); return; end;
for i=1:2:nargin
  fprintf('max(|%s-%s|) = ',inputname(i),inputname(i+1));
  if all(size(varargin{i})==size(varargin{i+1}))
    fprintf('%g\n',max(abs(varargin{i}(:)-varargin{i+1}(:))));
  else
    fprintf('dimension mismatch\n');
  end
end

