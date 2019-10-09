% [c,h] = fcontours(f,paramf,E,paramE,I,paramI,range[,Nc,Np])
% Contours of a 2D function with equality and inequality constraints
%
% Example usage: the following will plot Rosenbrock's function (with
% default parameter a = 100) in the rectangle [-2 2] x [-1 3], using a
% grid of 50x50 points and 80 contour lines:
%
%   fcontours(@Frosenbrock,{},{},{},{},{},[-2 2;-1 3],80,[50;50]);
%
% To get some contours around the unique minimum at (1,1), you need to give
% levels by hand:
%
% fcontours(@Frosenbrock,{},{},{},{},{},[-2 2;-1 3],[0:0.1:1 2:9 10:100:3000]);
%
% This plots a quadratic function f(x) = ½x'*A*x + b'*x + c in that same
% rectangle:
%
%   fcontours(@Fquad,{A,b,c},{},{},{},{},[-2 2;-1 3]);
%
% This plots Rosenbrock's function with that quadratic function as
% inequality constraint:
%
%   fcontours(@Frosenbrock,{},{},{},{@Fquad},{{A,b,c}},[-2 2;-1 3],80,[50;50]);
%
% The region where the inequality constraint is negative (nonnegative)
% appears in pink (white). The equality constraints appear as a black
% zero-level contour.
%
% In:
%   f: a handle to a 2D function.
%   paramf: cell array containing the parameters for f. Use {} to pass no
%      parameters, or to use f's default parameters.
%   E: a cell array {e1,...} of equality constraints. Each ei is a handle to
%      a function like f. Use {} if there are no equality constraints.
%   paramE: a cell array {p1,...} containing the parameters for e1,... (pi is
%      itself a cell array, like paramf). Use {} for pi to pass no parameters,
%      or to use pi's default parameters.
%   I, paramI: like E, paramE but for inequality constraints.
%   range: 2x2 matrix, containing rowwise the domain intervals of the
%      variables x1 and x2.
%   Nc: number of contour lines or, if a vector, the level values to use for
%      the contours (see Matlab help for `contour'). Default: up to Matlab.
%   Np: 2x1 list containing the number of grid points for each variable.
%      Default: [100;100].
% Out:
%   c, h: contour matrix and handle for f (see Matlab's help for "contour").
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2006 by Miguel A. Carreira-Perpinan

function [c,h] = fcontours(f,paramf,E,paramE,I,paramI,range,Nc,Np)

% ---------- Argument defaults ----------
if ~exist('Np','var') | isempty(Np) Np = [100;100]; end;
% ---------- End of "argument defaults" ----------

x = linspace(range(1,1),range(1,2),Np(1));
y = linspace(range(2,1),range(2,2),Np(2));
[X,Y] = meshgrid(x,y);

% Inequality constraints
if ~isempty(I)
  ZI = ones(size(X));
  for i=1:length(I)
    ZI = ZI.*(reshape(I{i}([X(:) Y(:)],paramI{i}{:}),size(X)) >= 0);
  end
  ZZI = repmat(1-ZI,[1 1 3]).*...
        repmat(reshape([1.00 0.88 0.88],1,1,3),[size(X) 1])...
        + repmat(ZI,[1 1 3]).*repmat(reshape([1 1 1],1,1,3),[size(X) 1]);
  image(x,y,ZZI);
end

set(gca,'DataAspectRatio',[1 1 1],'YDir','normal','Box','on');
hold on;
% Equality constraints
for i=1:length(E)
  ZE = reshape(E{i}([X(:) Y(:)],paramE{i}{:}),size(X));
  contour(X,Y,ZE,[0 0],'LineColor','k');
end
% $$$ % Inequality constraints
% $$$ for i=1:length(I)
% $$$   ZE = reshape(I{i}([X(:) Y(:)],paramI{i}{:}),size(X));
% $$$   contour(X,Y,ZE,[0 0],'LineColor','k');
% $$$ end
% Function f
Z = reshape(f([X(:) Y(:)],paramf{:}),size(X));
if ~exist('Nc','var') | isempty(Nc)
  [c,h] = contour(X,Y,Z);
  else
  [c,h] = contour(X,Y,Z,Nc);
end;
hold off;
grid on;
axis([range(1,:) range(2,:)]);

