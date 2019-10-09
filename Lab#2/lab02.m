% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Gaussian classifiers (D dimensions, K classes).
% We consider isotropic/diagonal/full covariance matrices ('I','D','F')
% which can also be shared across all classes ('i','d','f').


% ---------- 1D dataset: Gaussian clusters ----------

% Manual classes
gm.p = [2 1]'; gm.c = [0;2]; gm.S = [0.5;3];		% K=2 (for ROC)
%gm.p = [2 1 5]'; gm.c = [0;2;6]; gm.S = [0.5;3;2];	% K=3
%% Random classes
%K = 5; gm.p = rand(K,1); gm.c = sort(randn(K,1)); gm.S = rand(K,1); 

gm.type = 'I'; gm.p = gm.p/sum(gm.p); K = length(gm.p);
% Generate dataset: make sure each class has at least one data point
N = 100; [X,Y] = GMsample(N,gm);	% N = 1000;

% Fit Gaussian classifier and compute classification error and (if K=2)
% ROC curve on the training set
GMtype = {'I','i'}; lGM = length(GMtype); GM = cell(1,lGM);
ROCx = NaN(N+2,lGM); ROCy = ROCx; E = zeros(3,lGM); Eleg = cell(1,lGM);
for ii=1:lGM
  GM{ii} = GMclass(X,Y,GMtype{ii});
  [~,~,P] = GMpdf(X,GM{ii}); [e,~,Cn,L] = confumat(K,Y,P);
  E(1,ii) = e; E(2,ii) = Cn(2,1); E(3,ii) = Cn(1,1);
  if K==2
    [C,T,A] = roc(Y,P(:,1));
    ROCx(1:size(C,1),ii) = C(:,1); ROCy(1:size(C,1),ii) = C(:,2);
    Eleg{ii} =[GMtype{ii} '; \ast=' num2str(100*E(1,ii)) '%, AUC=' num2str(A)];
  end
end

% Plot p(x|C), p(C|x)
Nx = 1000; x = linspace(min(gm.c-4*gm.S),max(gm.c+4*gm.S),Nx)';
for ii=1:lGM
  [p,px_m,pm_x,pxm] = GMpdf(x,GM{ii});		% p(x), p(x|C), p(C|x), p(x,C)
  
  figure(ii); clf;
  set(gcf,'Name',['1D dataset; covariance type: ' GM{ii}.type]);
  subplot(2,1,1); h = plot(x,px_m,'-'); col = cell2mat(get(h,'Color'));
  hold on; plot(x,pxm,'-','LineWidth',3);
  scatter(X,-Y*0.09*max(px_m(:))/K,72,col(Y,:),'x'); hold off;
  axis([x([1 end])' max(px_m(:))*[-0.1 1]]);
  legend(h,num2str((1:K)')); xlabel('x'); ylabel('p(x|C), p(x,C)');
  title(['Classification error on training set: ' num2str(100*E(1,ii)) '%']);
  subplot(2,1,2); plot(x,pm_x,'-','LineWidth',3);
  hold on; scatter(X,-Y*0.09/K,72,col(Y,:),'x'); hold off;
  [~,J] = max(pm_x,[],2);
  hold on; scatter(x,1.05*ones(size(x)),100,col(J,:),'.'); hold off;
  xlabel('x'); ylabel('p(C|x)'); axis([x([1 end])' -0.1 1.09]);
end

% ROC curves on training set
% Note the ROC "curve" strictly is just the discrete collection of <=N points.
% It is not a continuous curve, nor does it have to reach (1,1) and (0,0).
if K==2
  figure(ii+1); set(gcf,'Name','1D dataset: ROC curves on training set'); clf;
  plot(ROCx,ROCy,'-',E(2,:),E(3,:),'r*'); daspect([1 1 1]); box on;
  legend(Eleg); xlabel('FP-rate'); ylabel('TP-rate');
end


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

% Fit Gaussian classifier
GMtype = {'I','i','D','d','F','f'}; lGM = length(GMtype); GM = cell(1,lGM);
for ii=1:lGM, GM{ii} = GMclass(X,Y,GMtype{ii}); end

% Plot
col = {'rgbkcmy','o+x.+sdv^<>ph'};	% colors and markers for points
COL = map50(K);				% colors for K cluster regions
t=linspace(0,2*pi,100)'; xy=[cos(t) sin(t)]; xx=[0 1;-1 0;0 -1;1 0]; % circle
% Grid of points for contour plots
lx1 = 100; lx2 = 100;
x1 = linspace(AX(1),AX(2),lx1); x2 = linspace(AX(3),AX(4),lx2);
[XX,YY] = meshgrid(x1,x2);

for ii=1:lGM
  
  % p(x), p(x|C), p(C|x), p(x,C)
  [p,px_m,pm_x,pxm] = GMpdf([XX(:) YY(:)],GM{ii});

  figure(ii+9); clf;
  set(gcf,'Name',['2D dataset; covariance type: ' GM{ii}.type]);
  set(gcf,'Position',[0 0 1280 310],'PaperPositionMode','auto');
  axes('Position',[0.04 0.075 0.30 0.9]); hold on;
  for k=1:K
    [k1,k2] = ind2sub([length(col{1}) length(col{2})],k);
    J = find(Y==k); plot(X(J,1),X(J,2),[col{1}(k1) col{2}(k2)],'MarkerSize',8);
    contour(x1,x2,reshape(px_m(:,k),lx1,lx2),30,col{1}(k1));	% p(x|C)
    plot(GM{ii}.c(k,1),GM{ii}.c(k,2),'>','MarkerSize',16,...	% mean|C
         'MarkerFaceColor',col{1}(k1));
  end
  hold off; daspect([1 1 1]); axis(AX); box on;
  xlabel('x_1'); ylabel('x_2'); title('p(x|C)');

  % Plot posterior probabilities as colormap
  axes('Position',[0.365 0.075 0.30 0.9]); hold on;
  C3 = pm_x*COL; C3(C3>1)=1; image(x1,x2,reshape(C3,lx2,lx1,3));
  plot(X(:,1),X(:,2),'k+');
  plot(GM{ii}.c(:,1),GM{ii}.c(:,2),'k*','MarkerSize',12);
  % Plot covariance matrix for each component as one standard deviation contour
  for m = 1:length(GM{ii}.p)
    switch GM{ii}.type
     case 'F', [V,D] = eig(GM{ii}.S(:,:,m));
     case 'D', [V,D] = eig(diag(GM{ii}.S(m,:)));
     case 'd', [V,D] = eig(diag(GM{ii}.S));
     case 'I', [V,D] = eig(GM{ii}.S(m)*eye(size(X,2)));
     case 'i', [V,D] = eig(GM{ii}.S*eye(size(X,2)));
    end
    xy1 = bsxfun(@plus,xy*(V*sqrt(D))',GM{ii}.c(m,:));
    yy = bsxfun(@plus,xx*(V*sqrt(D))',GM{ii}.c(m,:));
    plot(yy([1 3],1),yy([1 3],2),'k-',yy([2 4],1),yy([2 4],2),'k-',...
         xy1(:,1),xy1(:,2),'k-','LineWidth',2);
  end
  hold off; axis(AX); daspect([1 1 1]); box on;
  xlabel('x_1'); title('p(C|x)');

  axes('Position',[0.695 0.075 0.30 0.9]); hold on;
  [~,C] = max(pm_x,[],2); colormap(COL); image(x1,x2,reshape(C,lx2,lx1));
  plot(X(:,1),X(:,2),'k+');
  plot(GM{ii}.c(:,1),GM{ii}.c(:,2),'k*','MarkerSize',12);
  hold off; axis(AX); daspect([1 1 1]); box on;
  xlabel('x_1'); title('argmax_C{p(C|x)}');

end


% ---------- MNIST dataset ----------
load MNIST.mat; [N,D] = size(train_fea); DD = sqrt(D); Nt = size(test_fea,1);
train_fea = double(train_fea)/255; test_fea = double(test_fea)/255;
K = 10;

% Use random subset
I1 = randperm(N,1000); X = train_fea(I1,:); Y = train_gnd(I1);	% training
I2 = randperm(Nt,1000); Xt = test_fea(I2,:); Yt = test_gnd(I2);	% test
% If you use a subset of classes, make sure to relabel them as 1:K

% Fit Gaussian classifier
GMtype = {'I','i','D','d','F','f'}; lGM = length(GMtype); GM = cell(1,lGM);
for ii=1:lGM, GM{ii} = GMclass(X,Y,GMtype{ii}); end

% Plot
ii = 1;					% classifier (index into GM)
[p,px_m,pm_x,pxm] = GMpdf(Xt,GM{ii});	% p(x), p(x|C), p(C|x), p(x,C)
[e,C,Cn,L] = confumat(K,Yt,pm_x);	% error, conf. matrix, predicted class

% Plot class mean as images (they are the same for all GM types)
figure(20); set(gcf,'Name',['MNIST dataset; covariance type: ' GM{ii}.type]);
clf; colormap(gray(256));
for k=1:K
  subplot(1,K,k); imagesc(reshape(GM{ii}.c(k,:),DD,DD),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  title({['p(C_{' num2str(k) '})=' num2str(GM{ii}.p(k))],...
         ['mean(C_{' num2str(k) '}):']});
end

% Plot confusion matrix
figure(21);
set(gcf,'Name',...
        ['MNIST dataset confusion matrix; covariance type: ' GM{ii}.type]);
clf; colormap(parula(256)); imagesc(1:K,1:K,Cn,[0 1]); colorbar;  
axis image; set(gca,'XTick',1:K,'YTick',1:K);
xlabel('Predicted label'); ylabel('True label');
title(['Classification error: ' num2str(100*e) '%']);

% Plot a few images and the predicted p(C|x) as a histogram
J = ones(1,K); for k=1:K, J(1,k) = find(Yt==k,1); end	% one per class
%J = 1:length(Yt);					% all
%J = find(Yt~=L)';					% misclassified
figure(22);
set(gcf,'Name',['MNIST dataset examples; covariance type: ' GM{ii}.type]);
for i=J
  set(0,'CurrentFigure',22); clf; colormap(gray(256));
  subplot(1,2,1); bar(1:K,pm_x(i,:)); xlim([0.5 K+0.5]);
  title('p(C|x)'); xlabel('predicted label');
  subplot(1,2,2);
  imagesc(reshape(Xt(i,:),DD,DD),[0 1]); axis off; daspect([1 1 1]);
  title({['input image ' num2str(i)],['true label ' num2str(Yt(i))],...
        ['predicted label ' num2str(L(i))]});
  pause;
end


% Suggestions of things to try:
% - In general:
%   . Try other datasets.
%   . Try different Gaussian classifiers: 'I','i','D','d','F','f'.
%   . Change the training set size.
%   . Make the classes overlap more or less.
%   . Evaluate the classification error, ROC curve and confusion matrix on a
%     training and a test set.
% - For 1D datasets with K=2 try:
%   . Change the training set size: N = 100, 1000.
%   . Change the overlap between classes:
%     gm.p = [2 1]'; gm.c = [0;2]; gm.S = [0.5;3];	% medium overlap
%     gm.p = [2 1]'; gm.c = [0;1]; gm.S = [0.5;3];	% high overlap
%     gm.p = [2 1]'; gm.c = [0;5]; gm.S = [0.5;3];	% low  overlap
% - For 2D datasets:
%   . What is the form of the class boundaries (linear, quadratic...),
%     depending on the type of Gaussian classifier (eg 'f' vs 'F')?
%   . Manually set the class priors to the same value (1/K). How do the class
%     boundaries for the 'i' classifier look like?
% - For MNIST:
%   . Try using only 1s, 2s, 3s. Are the results better than using all classes?
%   . Compute the confusion matrix (for K>2) or ROC curve (for K=2).

