% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Demonstrate nonparametric methods:
% k-nearest-neighbor classifier, on 2D datasets.

% ---------- 2D dataset: Gaussian clusters, 2moons ----------

% 2moons
load 2moons.mat; [N,D] = size(X); K = max(unique(Y)); AX = [-2 3 -1.5 1.6];

%% Random Gaussian classes
%K = 5; gm.p = rand(K,1); gm.c = sort(randn(K,2),1); gm.S = zeros(2,2,K); 
%for k=1:K A = randn(2,2); gm.S(:,:,k) = A'*diag(rand(2,1))*A; end
%gm.type = 'F'; gm.p = gm.p/sum(gm.p); K = length(gm.p);
%% Generate dataset: make sure each class has at least one data point
%N = 500; [X,Y] = GMsample(N,gm);
%AX = [min(gm.c,[],1)-2 max(gm.c,[],1)+2]; AX = AX([1 3 2 4]);

% Plot
col = {'rgbkcmy','o+x.+sdv^<>ph'};	% colors and markers for points
COL = map50(K);				% colors for K cluster regions
% Grid of points for contour plots
lx1 = 100; lx2 = 100;
x1 = linspace(AX(1),AX(2),lx1); x2 = linspace(AX(3),AX(4),lx2);
[XX,YY] = meshgrid(x1,x2);

figure(1); set(gcf,'Name','2D dataset'); clf; hold on;
for k=1:K
  [k1,k2] = ind2sub([length(col{1}) length(col{2})],k);
  J = find(Y==k); plot(X(J,1),X(J,2),[col{1}(k1) col{2}(k2)],'MarkerSize',8);
end
hold off; daspect([1 1 1]); axis(AX); box on; xlabel('x_1'); ylabel('x_2');

figure(2); set(gcf,'Name','2D dataset: k-nearest-neighbor classifier'); clf;
for KK=1:10				% k parameter
  L = knn(X,Y,[XX(:) YY(:)],KK);
  hold on; colormap(COL); image(x1,x2,reshape(L,lx2,lx1));
  plot(X(:,1),X(:,2),'k+'); hold off;
  axis(AX); daspect([1 1 1]); box on; xlabel('x_1'); ylabel('x_2');
  title(['k-nearest-neighbor classifier, k = ' num2str(KK)]);
  pause
end


% Suggestions of things to try:
%
% - Other datasets:
%   . Different numbers of classes.
%   . Classes with different shapes and overlap.
%   . Try MNIST (use a small subset) and compare with the knn results in
%       http://yann.lecun.com/exdb/mnist

