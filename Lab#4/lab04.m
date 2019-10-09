% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Clustering algorithms: k-means, EM for Gaussian mixtures, mean-shift and
% connected-components with an e-ball graph.


% ---------- 2D toy dataset ----------

rng(1778);
x = linspace(-1,1,100); y = 1 - x.^2; X = [x(1:50);y(1:50)];
X = [X [x+1;0.3-y]]'; X = X + randn(size(X))/10; N = size(X,1); K = 2;

% 1. k-means with K clusters
K1 = 2; [C,L,E1] = kmeans(X,K1,'rndsubset');

% 2. EM training of a GM with K full-covariance components
K2 = 2; [gm,E2] = GMEM(X,K2,'F');

% 3. Mean-shift clustering
s3 = 0.3; [C3,L3] = mean_shift(X,s3);

% 4. Connected-components clustering with e-ball graph
e4 = 0.3; [C4,L4] = cc_eball(X,e4);


% ---------------------------------- Plots ----------------------------------

col = {'rgbkcmy','o+x.+sdv^<>ph'};	% colors and markers for points
COL = map50(K);				% colors for K cluster regions
t=linspace(0,2*pi,100)'; xy=[cos(t) sin(t)]; xx=[0 1;-1 0;0 -1;1 0]; % circle

% Grid of points for contour plots
lx1 = 100; lx2 = 100; ax = [-1.4 2.3 -1.2 1.4];
x1 = linspace(ax(1),ax(2),lx1); x2 = linspace(ax(3),ax(4),lx2);
[XX,YY] = meshgrid(x1,x2);

% Contour plots for EM / mean-shift: pdf p(x) and postprobs p(m|x)
[p,px_m,pm_x,pxm] = GMpdf([XX(:) YY(:)],gm);
kde.p = ones(N,1)/N; kde.c = X; kde.S = s3^2; kde.type = 'i';
kde_p = GMpdf([XX(:) YY(:)],kde);

% Dataset
figure(1); clf; plot(X(:,1),X(:,2),'b+'); axis(ax); daspect([1 1 1]); box on;

% k-means result
figure(2); plot(E1,'bo-');
xlabel('iteration'); ylabel('error'); title('k-means learning curve');
figure(3); clf; hold on;
plot(C(:,1),C(:,2),'k*','MarkerSize',12);
for k=1:K1
  [k1,k2] = ind2sub([length(col{1}) length(col{2})],k);
  J = find(L==k); plot(X(J,1),X(J,2),[col{1}(k1) col{2}(k2)]);
end
hold off; axis(ax); daspect([1 1 1]); box on;
title(['k-means: K = ' K1 ' clusters, error = ' num2str(E1(end))]);

% GM result
figure(4); plot(E2,'bo-');
xlabel('iteration'); ylabel('error'); title('EM learning curve');
figure(5); clf; hold on;
% Plot postprobs as colormap
COL3 = pm_x*COL; COL3(COL3>1)=1; image(x1,x2,reshape(COL3,lx2,lx1,3));
plot(X(:,1),X(:,2),'k+'); plot(gm.c(:,1),gm.c(:,2),'k*','MarkerSize',12);
% Plot covariance matrix for each component as one standard deviation contour
for m = 1:length(gm.p)
  switch gm.type
   case 'F', [V,D] = eig(gm.S(:,:,m));
   case 'D', [V,D] = eig(diag(gm.S(m,:)));
   case 'd', [V,D] = eig(diag(gm.S));
   case 'I', [V,D] = eig(gm.S(m)*eye(size(X,2)));
   case 'i', [V,D] = eig(gm.S*eye(size(X,2)));
  end
  xy1 = bsxfun(@plus,xy*(V*sqrt(D))',gm.c(m,:));
  yy = bsxfun(@plus,xx*(V*sqrt(D))',gm.c(m,:));
  plot(yy([1 3],1),yy([1 3],2),'k-',yy([2 4],1),yy([2 4],2),'k-',...
       xy1(:,1),xy1(:,2),'k-','LineWidth',2);
end
hold off; axis(ax); daspect([1 1 1]); box on;
title(['EM for a GM: K = ' num2str(length(gm.p)) ' clusters, log-lik = ' ...
       num2str(-E2(end)) ', p(k|x)']);

% Mean-shift result
figure(6); clf; contour(x1,x2,reshape(kde_p,lx1,lx2),30);
hold on;
plot(C3(:,1),C3(:,2),'k*','MarkerSize',12);
for k=1:size(C3,1)
  [k1,k2] = ind2sub([length(col{1}) length(col{2})],k);
  J = find(L3==k); plot(X(J,1),X(J,2),[col{1}(k1) col{2}(k2)]);
end
hold off; axis(ax); daspect([1 1 1]); box on;
title(['Mean-shift: \sigma = ' num2str(s3) ', ' ...
       num2str(size(C3,1)) ' clusters']);

% Connected-components result
figure(7); clf; gplot(sqdist(X)<(e4^2),X,'r-');		% plot e-ball graph
hold on; plot(X(:,1),X(:,2),'ko');
hold off; axis(ax); daspect([1 1 1]); box on;
title(['Connected-components: \epsilon-ball graph with \epsilon = ' ...
       num2str(e4)]);
figure(8); clf; hold on;
for k=1:C4
  [k1,k2] = ind2sub([length(col{1}) length(col{2})],k);
  J = find(L4==k); plot(X(J,1),X(J,2),[col{1}(k1) col{2}(k2)]);
end
hold off; axis(ax); daspect([1 1 1]); box on;
title(['Connected-components: \epsilon = ' num2str(e4) ', ' num2str(C4) ...
       ' clusters']);


% ---------- MNIST dataset ----------

load MNIST.mat; [N,D] = size(train_fea); DD = sqrt(D); Nt = size(test_fea,1);
train_fea = double(train_fea)/255; test_fea = double(test_fea)/255;

rng(1778);

% Subset to train and test on
%I1 = find(train_gnd==1);	% only 1s
I1 = randperm(N,1000); X = train_fea(I1,:); Y = train_gnd(I1);	% training
I2 = randperm(Nt,10); Xt = test_fea(I2,:); Yt = test_gnd(I2);	% test

% Train a GM with K full-covariance components
K = 10; [gm,E] = GMEM(X,K,'F');


% ---------------------------------- Plots ----------------------------------

figure(1); plot(E,'bo-');
xlabel('iteration'); ylabel('error'); title('EM learning curve');

% Plot the mean and weight of each GM component
figure(2); clf; colormap(gray(256));
set(gcf,'Name',['k-means on MNIST: means using K = ' num2str(K) ' clusters']);
k2 = ceil(sqrt(K)); k1 = ceil(K/k2);
for i=1:K
  subplot(k1,k2,i); imagesc(reshape(gm.c(i,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  title(['p(' num2str(i) ') = ' num2str(gm.p(i))]);
end

% Plot the posterior probabilities for the test set points
[p,~,pm_x] = GMpdf(Xt,gm);		% pdf p(x) and postprobs p(m|x)
w = 0.05;				% width of image relative to [0,1]
AX = [0.5 K+0.5 0 1];
figure(3); clf; colormap(gray(256));
set(gcf,'Name',['k-means on MNIST: component posterior probabilities p(k|x)']);
for i=1:size(Xt,1)
  subplot(1,2,1); imagesc(reshape(Xt(i,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  title(['Ground truth: ' num2str(Yt(i))]);
  subplot(1,2,2); bar(pm_x(i,:));
  axis(AX); xlabel('component index k'); ylabel('p(k|x)');
  P = get(gca,'Position');
  ax = P(3)/(AX(2)-AX(1)); bx = P(1) - ax*AX(1); % map coordinate -> figure
  ay = P(4)/(AX(4)-AX(3)); by = P(2) - ay*AX(3);
  for k=1:K
    axes('position',[ax*k+bx-w/2 ay*1.05+by-w/2 w w]);
    imagesc(reshape(gm.c(k,:),DD,DD),[0 1]); axis off; daspect([1 1 1]);
  end
  pause
end


% Suggestions of things to try:
%
% - User parameter of the clustering algorithms:
%   . k-means: different K values; different random initializations.
%   . EM for GM: different K values; different random initializations;
%     different types of GM (isotropic/diagonal/full covariances).
%   . Mean-shift: different s values.
%   . Connected-components with e-ball graph: different e values.
%
% - Different datasets, having:
%   . Different numbers of clusters (or with no clusters!).
%   . Different overlap between clusters.
%   . Clusters of different shapes.
%
% - MNIST:
%   . Train only on digit 1s (for example).
%   . Vary the size of the training set.
%   . Visualize the clusters using PCA or LDA (using the cluster assignments
%     as labels).

