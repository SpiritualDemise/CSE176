% [f,fX,E] = mlptrain(X,Y,f[,l,eta,B,tol,maxit,p]) Train MLP to map f(X) = Y
%
% MLP with one hidden layer of H sigmoidal units and linear output units,
% trained by (stochastic) gradient descent to minimise the least-squares
% regression error:
%   E(f) = sum^N_{n=1}{ |yn - f(xn)|² }
% where f(x) = W2.s(W1.x) and s() is a sigmoid, including biases implicitly.
% For batch gradient descent (GD), we stop iterating if one of these occurs:
% the error increases, the error decreases less than tol in relative error, or
% we reach maxit iterations. For SGD, we always run exactly maxit iterations
% (epochs).
%
% Notes:
% - It doesn't implement momentum at the moment.
% - Although one GD iteration should take about the same time as one SGD
%   epoch (regardless of the minibatch size B), in this Matlab implementation
%   an SGD epoch with small B takes much longer because it is not vectorised
%   (unlike a GD iteration).
%
% In:
%   X: NxL matrix, N L-dim data points rowwise.
%   Y: NxD matrix, N D-dim data points rowwise.
%   f: if a scalar, the #hiddens to use;
%      else, (struct) the MLP, initial values for the weights and biases.
%   l: (nonnegative scalar) weight decay parameter. Default: 0.
%   eta: backprop learning rate (gradient step size). Default: 1e-4.
%   B: the minibatch size in [1,N] for SGD (B=N gives batch gradient descent).
%      Default: N.
%   tol: minimum relative decrease in error to keep iterating. Default: 1e-6.
%   maxit: maximum number of iterations (epochs, for SGD). Default: 1e5.
%   p: 1 to print errors during training, 0 to not print. Default: 0.
% Out:
%   f: (struct) the MLP, with fields:
%      type='mlp'; weight matrices (incl. bias) layers 1/2: W1 (Hx(L+1)),
%      W2 (Dx(H+1)); weight decay parameter l.
%   fX: NxD matrix, f(X).
%   E: sequence of errors during the optimisation, where E(1) is the error
%      with the initial weights.

% Copyright (c) 2015 by Miguel A. Carreira-Perpinan

function [f,fX,E] = mlptrain(X,Y,f,l,eta,B,tol,maxit,p)

[N,L] = size(X); D = size(Y,2);

% ---------- Argument defaults ----------
if ~exist('l','var') | isempty(l) l = 0; end;
if ~exist('eta','var') | isempty(eta) eta = 1e-4; end;
if ~exist('B','var') | isempty(B) B = N; end;
if ~exist('tol','var') | isempty(tol) tol = 1e-6; end;
if ~exist('maxit','var') | isempty(maxit) maxit = 1e5; end;
if ~exist('p','var') | isempty(p) p = 0; end;
% ---------- End of "argument defaults" ----------

if ~isstruct(f)
  H = f; clear f;
  % Initialise weights and biases to small random values
  f.W1 = randn(H,L+1)/10; f.W2 = randn(D,H+1)/10;
end;
f.type = 'mlp'; f.l = l;

fX = mlp(X,f); e = (fX-Y)'; E = zeros(1,maxit+1); E(1) = e(:)'*e(:);
if p fprintf('\r  mlptrain iteration: %3d; error: %f',0,E(1)); end

if B==N							% Batch GD
  
  cont = (maxit>=1); i = 0;
  while cont
    
    % Gradients
    [z,dz] = sigmoid(f.W1*[X';ones(1,N)]); gW2 = [e*z' sum(e,2)] + l*f.W2;
    ee = dz.*(f.W2(:,1:end-1)'*e); gW1 = [ee*X sum(ee,2)] + l*f.W1;
    
    % Gradient step
    f.W1 = f.W1 - 2*eta*gW1; f.W2 = f.W2 - 2*eta*gW2;
    
    % New errors
    fX = mlp(X,f); e = (fX-Y)'; E(i+2) = e(:)'*e(:);
    
    if p fprintf('\r  mlptrain iteration: %3d; error: %f',i+1,E(i+2)); end
    i = i + 1; cont = ((E(i)-E(i+1))>tol*E(i)) & (i<maxit);
    
  end
  E = E(1:i+1);
  if E(end)>E(end-1) fprintf(' ***overshoot'); end;
  
else							% SGD
  
  for i=2:maxit+1
    NN = randperm(N);					% Shuffle points
    n0 = 1;
    for n=1:ceil(N/B)
      
      I = NN(n0:min(N,n0+B-1));				% Minibatch
      
      % Error for minibatch
      fX2 = mlp(X(I,:),f); e2 = (fX2-Y(I,:))';
      
      % Gradients for minibatch
      [z,dz] = sigmoid(f.W1*[X(I,:)';ones(1,B)]); gW2=[e2*z' sum(e2,2)]+l*f.W2;
      e1 = dz.*(f.W2(:,1:end-1)'*e2); gW1 = [e1*X(I,:) sum(e1,2)] + l*f.W1;
      
      % Gradient step for minibatch
      f.W1 = f.W1 - 2*eta*gW1; f.W2 = f.W2 - 2*eta*gW2;
      
      n0 = n0+B;
    
    end
    if p | (nargout>2) fX = mlp(X,f); e = (fX-Y)'; E(i) = e(:)'*e(:); end
    if p fprintf('\r  mlptrain iteration: %3d; error: %f',i-1,E(i)); end
    
  end

  if nargout>1 fX = mlp(X,f); end
  
end

if p fprintf('.\n'); end

