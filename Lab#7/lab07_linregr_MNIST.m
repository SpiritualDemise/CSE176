% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Demonstrate (stochastic) gradient descent in MNIST with:
% - Linear regression: f(x) = W.x + w
%   trained with least-squares error.

load MNIST.mat

I = find(train_gnd==1); Iv = find(test_gnd==1);			% only 1s
X = double(train_fea(I(1:1000),:))/255; X = X + 1e-3*randn(size(X));
Xv = double(test_fea(Iv(1:1000),:))/255; Xv = Xv + 1e-3*randn(size(Xv));
[N,D] = size(X); Nv = size(Xv,1);

% Ground-truth linear mapping and noise
F.type = 'linf';
% - Random linear mapping
% E = 10; F.W = randn(E,D)/10+0.02; F.w = randn(E,1)/10; s = 0.5;
% - Flip & clip
ox = 5; oy = 8; s = 0.001;
Z = reshape(1:D,sqrt(D),sqrt(D)); Z = Z(10:10+ox-1,15:15+oy-1); Z = Z(:);
E = length(Z); F.W=zeros(E,D); F.W(sub2ind([E D],1:E,Z'))=1; F.w=zeros(E,1);

% Construct output labels
Y = linf(X,F); Y = Y + s*randn(size(Y));
Yv = linf(Xv,F); Yv = Yv + s*randn(size(Yv));

% Linear regression, exactly by solving the normal equations
f = linfexact(X,Y);
e = Y - linf(X,f); ef = e(:)'*e(:)/2;			% Optimal error

% Initial weights for both GD and SGD
g0.W = randn(E,D)/1e3; g0.w = randn(E,1)/1e3; g0.type = 'linf';
maxit = 10;						% Max # iterations

% Linear regression, iteratively with gradient descent
eta = 1e-5;						% Step size
[GDg,GDe] = linfgd(X,Y,Xv,Yv,eta,maxit,g0); gd = GDg{end,end};

% Linear regression, iteratively with stochastic gradient descent
Eta = 5e-3;						% Step size
B = 10;							% Minibatch size
[SGDg,SGDee] = linfsgd(X,Y,Xv,Yv,Eta,maxit,B,g0); sgd = SGDg{end,end};


% ---------------------------------- Plots ----------------------------------

% Put iterates in an array for plotting
GDe = diag([N Nv].^(-1))*GDe;
SGDee(1,:,:) = SGDee(1,:,:)/N; SGDee(2,:,:) = SGDee(2,:,:)/Nv;
SGDe = squeeze(SGDee(:,end,:));

% Learning curves: error over iterations
figure(2); clf;
set(gcf,'Name','MNIST dataset; linear regression by least-squares error');
subplot(2,1,1);
plot([0 maxit],[ef ef],'k-',0:maxit,GDe(1,:),'r-',0:maxit,SGDe(1,:),'m-',...
     0:maxit,GDe(2,:),'r--',0:maxit,SGDe(2,:),'m--');
legend('optimal f','GD f (training)','SGD f (training)',...
       'GD f (validation)','SGD f (validation)');
hold on; box on; xlabel('iteration (epoch)'); ylabel('error');
axis([0 maxit 0 max(GDe(1,1),SGDe(1,1))]);
title(['\eta_{GD} = ' num2str(eta) ', \eta_{SGD} = ' num2str(Eta) ...
       ', B = ' num2str(B)]);
h3 = plot(0,GDe(1,1),'ro',0,SGDe(1,1),'mo');
subplot(2,1,2); z1 = SGDee(1,:,:); z2 = SGDee(2,:,:);
plot([0 maxit],[ef ef],'k-',...
     linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z1(:);NaN],'m-',...
     linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z2(:);NaN],'m:');
legend('optimal f','SGD f (training)','SGD f (validation)');
hold on; box on; axis([0 maxit 0 max(GDe(1,1),SGDe(1,1))]);
xlabel('iteration (epoch)'); ylabel('error after each minibatch');
h4 = plot(0,SGDe(1,1),'mo');

% Plot of sample images and their outputs
idx = 1:10; lidx = length(idx); XX = Xv(idx,:); YY = Yv(idx,:);	% first 10 img
FXX = linf(XX,F); fXX = linf(XX,f); gdXX = linf(XX,gd); sgdXX = linf(XX,sgd);
figure(3); clf; colormap(gray(256));
set(gcf,'Name','MNIST dataset: sample images and their outputs');
for i=1:lidx
  subplot(5,lidx,i);
  imagesc(reshape(XX(i,:),sqrt(D),sqrt(D)),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  if i==1 ylabel('x'); end
  subplot(5,lidx,i+lidx); imagesc(reshape(FXX(i,:),ox,oy),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  if i==1 ylabel('true f(x)'); end
  subplot(5,lidx,i+2*lidx); imagesc(reshape(fXX(i,:),ox,oy),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  if i==1 ylabel('optimal f(x)'); end
  subplot(5,lidx,i+3*lidx); imagesc(reshape(gdXX(i,:),ox,oy),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  if i==1 ylabel('GD f(x)'); end
  subplot(5,lidx,i+4*lidx); imagesc(reshape(sgdXX(i,:),ox,oy),[0 1]);
  set(gca,'XTick',[],'YTick',[]); axis image;
  if i==1 ylabel('SGD f(x)'); end
end

% Plot of mapping coefficients
figure(4); clf; colormap(parula(256));
set(gcf,'Name','MNIST dataset: mapping coefficients');
subplot(4,1,1); imagesc([F.W NaN(E,1) F.w]);
set(gca,'XTick',[],'YTick',[]); axis image; title('true f(x) = A.x + b');
subplot(4,1,2); imagesc([f.W NaN(E,1) f.w]);
set(gca,'XTick',[],'YTick',[]); axis image; title('optimal f(x)');
subplot(4,1,3); imagesc([gd.W NaN(E,1) gd.w]);
set(gca,'XTick',[],'YTick',[]); axis image; title('GD f(x)');
subplot(4,1,4); imagesc([sgd.W NaN(E,1) sgd.w]);
set(gca,'XTick',[],'YTick',[]); axis image; title('SGD f(x)');

% Suggestions of things to try: as for lab06_linregr.m

