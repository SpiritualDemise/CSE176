% This code demonstrates Laplacian K-modes on the 5-spirals dataset (using a
% single bandwidth value s, that is, without homotopy).

rng(0);

% Generate the 5-spirals dataset
K=5; nn=400; a=pi/4; b=4*pi; t=((b-a)*(linspace(0,1,nn).^0.65)+a)';
X = []; label=[];
for j=0:K-1
  X1 = [t.*sin(t+2*pi*j/K) t.*cos(t+2*pi*j/K)];
  X = [X; X1]; label=[label; (j+1)*ones(nn,1)];
end
X = X+randn(size(X))*0.1; N=size(X,1);
figure(1); plot(X(:,1),X(:,2),'bo');
daspect([1 1 1]); ax = [-15 15 -15 15]; axis(ax); title('data');

% Set hyperparameters (K, s, lambda) and build graph and Laplacian
K = 5; s = 0.3; lambda = 2;
knn = 5; W = gaussaff(X,{'k',knn},s); d = sum(W,2); Laplacian = diag(d)-W;

% Use K-means for initialization
[C,label] = kmeans(X,K);
Z = zeros(N,K); Z(sub2ind([N,K],(1:N)',label)) = 1;

% Run Laplacian K-modes
[C,Z] = lapkmodes(X,K,s,lambda,Laplacian,[],[],C,Z);

% Plot clustering
COLORSX = {'bo','ro','go','co','mo','ko'};
COLORSC = {'bo','ro','go','co','mo','ko'};
figure(2); clf; hold on;
[confidence,IDX] = max(Z,[],2);
% Plot each cluster.
for k=1:K
  idx = find(IDX==k);  plot(X(idx,1),X(idx,2),COLORSX{k});
  plot(C(k,1),C(k,2),COLORSC{k},'MarkerSize',10,'MarkerFaceColor',COLORSC{k}(1));
end
daspect([1 1 1]); axis(ax); set(gca,'xtick',[],'ytick',[]); box on;
title('hard clustering');

% Plot KDE.
figure(3); clf; hold on;
for j=1:K   % Plot normalized KDE for each cluster.
  t1 = linspace(ax(1),ax(2),150);   t2 = linspace(ax(3),ax(4),150);
  [Y1,Y2] = meshgrid(t1,t2);   Y = [Y1(:),Y2(:)];
  pdf = exp(-sqdist(Y,X)/(2*s^2))*Z(:,j)/sum(Z(:,j))/(sqrt(2*pi)*s)^2;
  pdf = reshape(pdf,150,150);
  contour(Y1,Y2,pdf,10,COLORSX{j}(1));
end
daspect([1 1 1]); axis(ax); set(gca,'xtick',[],'ytick',[]); box on;
title('kde');

% Plot soft assignment using mixture of colors.
COLORSRGB = [[0 0 1];[1 0 0];[0 1 0];[0 1 1];[1 0 1];[0 0 0]];
figure(4); clf; hold on;
for j=1:size(X,1)
  cc=Z(j,:)*COLORSRGB(1:K,:);
  cc(cc<0)=0; cc(cc>1)=1;
  plot(X(j,1),X(j,2),'o','Color',cc);
end
daspect([1 1 1]); axis(ax); set(gca,'xtick',[],'ytick',[]); box on;
title('soft assignment');

