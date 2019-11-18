% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Demonstrate (stochastic) gradient descent in MNIST with:
% - Binary classification: logistic regression: f(x) = s(w'.x+b)
%   trained with maximum likelihood and least-squares error.

load MNIST.mat
rng(1779);

label = [3 5];					% class 0 = 3s, class 1 = 5s
%label = [1 7];					% class 0 = 1s, class 1 = 7s
I = find(train_gnd==label(1) | train_gnd==label(2));
Iv = find(test_gnd==label(1) | test_gnd==label(2));
I = I(randperm(length(I),300)); Iv = Iv(randperm(length(Iv),1000)); % subset
X = double(train_fea(I,:))/255; Xv = double(test_fea(Iv,:))/255;
Y = train_gnd(I,:); Y = Y==label(2); Yv = test_gnd(Iv,:); Yv = Yv==label(2);
[N,D] = size(X); DD = sqrt(D); Nv = size(Xv,1);

o = 0;							% max-llh
%o = 1;							% lsq-err
% GD/SGD step size for each objective function
if ~o
  eta = 7e-4;						% GD step size
  Eta = 5e-2;						% SGD step size
else
  eta = 2e-3;						% GD step size
  Eta = 5e-2;						% SGD step size
end

% Initial weights for both GD and SGD
g0.W = randn(1,D)/1e3; g0.w = randn/1e3; g0.type = 'slinf';
maxit = 10;						% Max # iterations

% Linear regression, iteratively with gradient descent
[GDg,GDe,GDc] = slinfgd(X,Y,Xv,Yv,o,eta,maxit,g0);

% Linear regression, iteratively with stochastic gradient descent
B = 1;							% Minibatch size
[SGDg,SGDee,SGDcc] = slinfsgd(X,Y,Xv,Yv,o,Eta,maxit,B,g0);


% ---------------------------------- Plots ----------------------------------

% Put iterates in an array for plotting
GDw = zeros(D+1,maxit+1); for i=1:maxit+1, GDw(:,i) = [GDg{i}.W GDg{i}.w]'; end
SGDww = zeros(D+1,ceil(N/B),maxit+1);			% Minibatch iterates
SGDww(:,:,1) = NaN; SGDww(:,1,1) = [SGDg{1,1}.W SGDg{1,1}.w]';
for i=1:maxit+1
  for n=1:ceil(N/B), SGDww(:,n,i+1) = [SGDg{n,i}.W SGDg{n,i}.w]'; end
end
SGDw = squeeze(SGDww(:,end,:));				% Iterates

% Objective function values, per data point
GDe = diag([N Nv].^(-1))*GDe;
SGDee(1,:,:) = SGDee(1,:,:)/N; SGDee(2,:,:) = SGDee(2,:,:)/Nv;
SGDe = squeeze(SGDee(:,end,:)); SGDc = squeeze(SGDcc(:,end,:));

objfcn = {'cross-entropy','least-squares error'};

% Learning curves: error (cross-entropy or least-squares) over iterations
figure(1); clf;
set(gcf,'Name',['MNIST dataset; logistic regression by ' objfcn{o+1}]);
subplot(2,1,1);
plot(0:maxit,GDe(1,:),'r-',0:maxit,SGDe(1,:),'m-',...
     0:maxit,GDe(2,:),'r--',0:maxit,SGDe(2,:),'m--');
legend('GD f (training)','SGD f (training)',...
       'GD f (validation)','SGD f (validation)');
hold on; box on; xlabel('iteration (epoch)'); ylabel(objfcn{o+1});
axis([0 maxit min([GDe(:);SGDee(:)]) max(GDe(1,1),SGDe(1,1))]);
title(['\eta_{GD} = ' num2str(eta) ', \eta_{SGD} = ' num2str(Eta) ...
       ', B = ' num2str(B)]);
h3 = plot(0,GDe(1,1),'ro',0,SGDe(1,1),'mo');
subplot(2,1,2); z1 = SGDee(1,:,:); z2 = SGDee(2,:,:);
plot(linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z1(:);NaN],'m-',...
     linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z2(:);NaN],'m:');
legend('SGD f (training)','SGD f (validation)');
hold on; box on; 
axis([0 maxit min([GDe(:);SGDee(:)]) max(GDe(1,1),SGDe(1,1))]);
xlabel('iteration (epoch)');
ylabel({objfcn{o+1},'after each minibatch'});
h4 = plot(0,SGDe(1,1),'mo');
print('-dpng','-r0','A0.png');

% Learning curves: classification error over iterations
figure(2); clf;
set(gcf,'Name',['MNIST dataset; logistic regression by ' objfcn{o+1}]);
subplot(2,1,1);
plot(0:maxit,GDc(1,:),'r-',0:maxit,SGDc(1,:),'m-',...
     0:maxit,GDc(2,:),'r--',0:maxit,SGDc(2,:),'m--');
legend('GD f (training)','SGD f (training)',...
       'GD f (validation)','SGD f (validation)');
hold on; box on; xlabel('iteration (epoch)'); ylabel('classification error');
axis([0 maxit 0 max(GDc(1,1),SGDc(1,1))]);
title(['\eta_{GD} = ' num2str(eta) ', \eta_{SGD} = ' num2str(Eta) ...
       ', B = ' num2str(B)]);
h3 = plot(0,GDc(1,1),'ro',0,SGDc(1,1),'mo');
subplot(2,1,2); z1 = SGDcc(1,:,:); z2 = SGDcc(2,:,:);
plot(linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z1(:);NaN],'m-',...
     linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z2(:);NaN],'m:');
legend('SGD f (training)','SGD f (validation)');
hold on; box on; 
axis([0 maxit 0 max(GDc(1,1),SGDc(1,1))]);
xlabel('iteration (epoch)');
ylabel({'classification error','after each minibatch'});
h4 = plot(0,SGDc(1,1),'mo');
print('-dpng','-r0','B0.png');

% Plot a few images and the output of the trained logistic regression
[~,J]=sort(abs(Yv-slinf(Xv,GDg{end})),'descend'); J=J(1:10)'; 	% hardest ones
%J = 1:10;							% first ones
%J = 1:length(Yv);						% all
figure(3);
set(gcf,'Name',['MNIST dataset: sample images and their outputs; using'...
                objfcn{o+1}]);
for i=J
  set(0,'CurrentFigure',3); clf; colormap(gray(256));
  subplot(2,2,1);
  z = slinf(Xv(i,:),GDg{end}); bar(0:1,[1-z z]); xlim([-0.5 1.5]);
  title(['GD: \sigma(w^Tx+w_0) = ' num2str(z)]); xlabel('predicted label');
  set(gca,'Xtick',[0;1],'XtickLabel',num2str(label'));
  subplot(2,2,2);
  imagesc(reshape(Xv(i,:),DD,DD),[0 1]); axis off; daspect([1 1 1]);
  title({['input image ' num2str(Iv(i))],...
         ['true label ' num2str(label(1+Yv(i)))],...
        ['predicted label ' num2str(label(1+(z>0.5))) ' with GD']});
  subplot(2,2,3);
  z = slinf(Xv(i,:),SGDg{end}); bar(0:1,[1-z z]); xlim([-0.5 1.5]);
  title(['SGD: \sigma(w^Tx+w_0) = ' num2str(z)]); xlabel('predicted label');
  set(gca,'Xtick',[0;1],'XtickLabel',num2str(label'));
  subplot(2,2,4);
  imagesc(reshape(Xv(i,:),DD,DD),[0 1]); axis off; daspect([1 1 1]);
  title({['input image ' num2str(Iv(i))],...
         ['true label ' num2str(label(1+Yv(i)))],...
         ['predicted label ' num2str(label(1+(z>0.5))) ' with SGD']});
print('-dpng','-r0',['C' num2str(i) '.png']);
end

% Plot of logistic regression weights (not the bias)
figure(4); clf; colormap(parula(256));
set(gcf,'Name',['MNIST dataset: logistic regression weights; using'...
                objfcn{o+1}]);
z1=GDw(1:D,:); z2=SGDww(1:D,:,2:end); z1=[z1(:);z2(:)]; r = [min(z1) max(z1)];
h = zeros(1,4); for j=1:4, h(j) = subplot(2,2,j); end
axes('position',[0.05 0.58 0.1 0.33]); title('colorbar');
imagesc(linspace(r(1),r(2),256)',r); daspect([1 50 1]);
set(gca,'XTick',[],'YTick',[1 128 256],'YTickLabel',num2str([r(2);0;r(1)],2));
for i=1:maxit+1
  set(0,'CurrentFigure',4);
  axes(h(1)); cla; imagesc(reshape(GDg{i}.W,DD,DD),r);
  set(gca,'XTick',[],'YTick',[]); axis image;
  title(['GD w, iteration ' num2str(i-1)]);
  axes(h(3)); cla; imagesc(reshape(GDg{i}.W,DD,DD));
  set(gca,'XTick',[],'YTick',[]); axis image; title('rescaled image');
  switch i	% which minibatches to plot, to keep the animation short
   case 1, Blist = 1;
   case 2, Blist = [1:20 30:10:ceil(N/B)];
   otherwise, Blist = 1:10:ceil(N/B);
  end
  Blist = Blist(Blist<=ceil(N/B));
  for n=Blist
    axes(h(2)); cla; imagesc(reshape(SGDg{n,i}.W,DD,DD),r);
    set(gca,'XTick',[],'YTick',[]); axis image;
    title(['SGD w, iteration ' num2str(i-1) ', minibatch ' num2str(n)]);
    axes(h(4)); cla; imagesc(reshape(SGDg{n,i}.W,DD,DD));
    set(gca,'XTick',[],'YTick',[]); axis image; title('rescaled image');
print('-dpng','-r0',['D' num2str(i) '_' num2str(n) '.png']);
  end
end


% Suggestions of things to try:
% - Overfitting is more likely to occur with difficult problems, eg 3s vs 5s.

