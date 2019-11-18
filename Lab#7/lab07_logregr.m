% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Demonstrate (stochastic) gradient descent in 1D and 2D datasets with:
% - Binary classification: logistic regression: f(x) = s(w'.x+b)
%   trained with maximum likelihood and least-squares error.

% 1D dataset: Gaussian clusters
rng(1778);
D = 1;
gm.p = [2 1]'; gm.c = [0;2]; gm.S = [0.5;3];			% manual clsses
%gm.p = rand(2,1); gm.c = sort(randn(2,D)); gm.S = rand(2,D);	% random clsses
gm.type = 'I'; gm.p = gm.p/sum(gm.p);
% Generate dataset: training & validation
N = 100; [X,Y] = GMsample(N,gm); Y(Y==2) = 0;
Nv = 140; [Xv,Yv] = GMsample(Nv,gm); Yv(Yv==2) = 0;

o = 0;							% max-llh
%o = 1;							% lsq-err
% Ranges for contour plots and GD/SGD step size for each objective function
if ~o
  AX = {[-3.5 1.5;-0.5 3.5],[-1.2 -0.6;1.1 1.6]};
  eta = 4.5e-3;						% GD step size
  Eta = 5e-2;						% SGD step size
else
  AX = {[-6 1;-0.1 6],[-2 -1.5;2.2 2.6]};		
  eta = 2e-1;						% GD step size
  Eta = 3e-1;						% SGD step size
end

% Initial weights for both GD and SGD
g0.W = randn(1,D)/1e3; g0.w = randn/1e3; g0.type = 'slinf';
maxit = 100;						% Max # iterations

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
Objfcn = {@Fxent,@Flsqerr};

% Dataset and logistic regressor estimates
Nx = 1000; x = linspace(min(gm.c-4*gm.S),max(gm.c+4*gm.S),Nx)';
GDSx = slinf(x,GDg{1}); SGDSx = slinf(x,SGDg{1,1});
col = [0 0 1;0 0.5 0];
figure(1); clf; hold on;
set(gcf,'Name',['1D dataset; logistic regression by ' objfcn{o+1}]);
h = scatter(X,Y,72,col(Y+1,:),'o');
hv = scatter(Xv,Yv-0.05*(2*Yv-1),72,col(Yv+1,:),'+');
h1 = plot(x,GDSx,'r-','LineWidth',2);
h2 = plot(x,SGDSx,'m-','LineWidth',2);
h1a = plot(x([1 end]),GDw(1,1)*x([1 end])+GDw(2,1),'r--');
h2a = plot(x([1 end]),SGDw(1,1)*x([1 end])+SGDw(2,1),'m--');
legend([h;hv;h1;h2],'training','validation','GD f','SGD f');
scatter(X,-0.04-(Y+1)*0.03,36,col(Y+1,:),'o');
text(x(1)*[1;1],-0.04-[1;2]*0.03,...
     ['class 1 \rightarrow';'class 2 \rightarrow'],...
     'HorizontalAlignment','right');
GDJ = GDSx > 0.5; SGDJ = SGDSx > 0.5;
h1b = scatter(x,1.05*ones(size(x)),100,col(GDJ+1,:),'.');
h2b = scatter(x,1.08*ones(size(x)),100,col(SGDJ+1,:),'.');
text(x(1)*[1;1],[1.05;1.08],[' GD \rightarrow';'SGD \rightarrow'],...
     'HorizontalAlignment','right');
box on; xlabel('x'); ylabel('\sigma(x)'); axis([x([1 end])' -0.1 1.1]);
title('iteration (epoch): 0');

% Learning curves: error (cross-entropy or least-squares) over iterations
figure(2); clf;
set(gcf,'Name',['1D dataset; logistic regression by ' objfcn{o+1}]);
subplot(2,1,1);
plot(0:maxit,GDe(1,:),'r-',0:maxit,SGDe(1,:),'m-',...
     0:maxit,GDe(2,:),'r--',0:maxit,SGDe(2,:),'m--');
legend('GD f (training)','SGD f (training)',...
       'GD f (validation)','SGD f (validation)');
hold on; box on; xlabel('iteration (epoch)'); ylabel(objfcn{o+1});
axis([0 maxit min([GDe(:);SGDee(:)]) max(GDe(1,1),SGDe(1,1))]);
title(['\eta_{GD} = ' num2str(eta) ', \eta_{SGD} = ' num2str(Eta) ...
       ', B = ' num2str(B)]);
h3a = plot(0,GDe(1,1),'ro',0,SGDe(1,1),'mo');
subplot(2,1,2); z1 = SGDee(1,:,:); z2 = SGDee(2,:,:);
plot(linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z1(:);NaN],'m-',...
     linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z2(:);NaN],'m:');
legend('SGD f (training)','SGD f (validation)');
hold on; box on; 
axis([0 maxit min([GDe(:);SGDee(:)]) max(GDe(1,1),SGDe(1,1))]);
xlabel('iteration (epoch)');
ylabel({objfcn{o+1},'after each minibatch'});
h3b = plot(0,SGDe(1,1),'mo');

% Learning curves: classification error over iterations
figure(3); clf;
set(gcf,'Name',['1D dataset; logistic regression by ' objfcn{o+1}]);
subplot(2,1,1);
plot(0:maxit,GDc(1,:),'r-',0:maxit,SGDc(1,:),'m-',...
     0:maxit,GDc(2,:),'r--',0:maxit,SGDc(2,:),'m--');
legend('GD f (training)','SGD f (training)',...
       'GD f (validation)','SGD f (validation)');
hold on; box on; xlabel('iteration (epoch)'); ylabel('classification error');
axis([0 maxit 0 max(GDc(1,1),SGDc(1,1))]);
title(['\eta_{GD} = ' num2str(eta) ', \eta_{SGD} = ' num2str(Eta) ...
       ', B = ' num2str(B)]);
h4a = plot(0,GDc(1,1),'ro',0,SGDc(1,1),'mo');
subplot(2,1,2); z1 = SGDcc(1,:,:); z2 = SGDcc(2,:,:);
plot(linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z1(:);NaN],'m-',...
     linspace(0,maxit+1,(maxit+1)*ceil(N/B)+1),[z2(:);NaN],'m:');
legend('SGD f (training)','SGD f (validation)');
hold on; box on; 
axis([0 maxit 0 max(GDc(1,1),SGDc(1,1))]);
xlabel('iteration (epoch)');
ylabel({'classification error','after each minibatch'});
h4b = plot(0,SGDc(1,1),'mo');

% Contour plots: error function and iterates
for j=1:length(AX)
  figure(2+2*j); clf;
  set(gcf,'Name',['1D dataset; logistic regression by: ' objfcn{o+1}]);
  fcontours(Objfcn{o+1},{g0,X,Y},{},{},{},{},AX{j},50,[50;50]);
  plotseq(GDw','r-'); plotseq(SGDw','m-');
  hold on; xlabel('w'); ylabel('w_0');
  title(['GD/SGD iterations; \eta_{GD} = ' num2str(eta) ', \eta_{SGD} = '...
         num2str(Eta) ', B = ' num2str(B)]);
  hh1{j}.a = plotseq(GDw(:,1)','ro'); hh1{j}.b = plotseq(SGDw(:,1)','mo');
  figure(3+2*j); clf;
  set(gcf,'Name',['1D dataset; logistic regression by: ' objfcn{o+1}]);
  fcontours(Objfcn{o+1},{g0,X,Y},{},{},{},{},AX{j},50,[50;50]);
  hold on; xlabel('w'); ylabel('w_0');
  title(['SGD minibatches; \eta_{SGD} = ' num2str(Eta) ', B = ' num2str(B)]);
  hh2{j}.a = plotseq(GDw(:,1)','ro'); hh2{j}.b = plotseq(SGDw(:,1)','mo');
end

% Plots over iterates
for i=2:maxit+1
  pause;
  set(0,'CurrentFigure',1);
  delete(h1); delete(h2); delete(h1a); delete(h2a); delete(h1b); delete(h2b);
  GDSx = slinf(x,GDg{i}); SGDSx = slinf(x,SGDg{1,i});
  h1 = plot(x,GDSx,'r-','LineWidth',2);
  h2 = plot(x,SGDSx,'m-','LineWidth',2);
  h1a = plot(x([1 end]),GDw(1,i)*x([1 end])+GDw(2,i),'r--');
  h2a = plot(x([1 end]),SGDw(1,i)*x([1 end])+SGDw(2,i),'m--');
  GDJ = GDSx > 0.5; SGDJ = SGDSx > 0.5;
  h1b = scatter(x,1.05*ones(size(x)),100,col(GDJ+1,:),'.');
  h2b = scatter(x,1.08*ones(size(x)),100,col(SGDJ+1,:),'.');
  title(['iteration (epoch): ' num2str(i-1)]);
  set(0,'CurrentFigure',2); delete(h3a); delete(h3b);
  subplot(2,1,1); h3a = plot(i-1,GDe(1,i),'ro',i-1,SGDe(1,i),'mo');
  subplot(2,1,2); h3b = plot(i-1,SGDe(1,i),'mo');
  set(0,'CurrentFigure',3); delete(h4a); delete(h4b);
  subplot(2,1,1); h4a = plot(i-1,GDc(1,i),'ro',i-1,SGDc(1,i),'mo');
  subplot(2,1,2); h4b = plot(i-1,SGDc(1,i),'mo');
  for j=1:length(AX)
    set(0,'CurrentFigure',2+2*j); delete(hh1{j}.a); delete(hh1{j}.b);
    hh1{j}.a = plotseq(GDw(:,i)','ro'); hh1{j}.b = plotseq(SGDw(:,i)','mo');
    set(0,'CurrentFigure',3+2*j); delete(hh2{j}.a); delete(hh2{j}.b);
    hh2{j}.a = plotseq(GDw(:,i)','ro'); hh2{j}.b = plotseq(SGDww(:,:,i)','m-');
  end
  drawnow;
end



% Suggestions of things to try:
% - Consider a training set that is linearly separable vs one that is not
%   linearly separable. How does the objective function (maximum likelihood
%   or least-squares error) look like in each case, and why? How does this
%   affect the optimal parameters? To see the difference easily, use the
%   following setup:
%   . A small training set such as this:
%     * Linearly separable: N = 4; X = (0:3)'; Y = [1 1 0 0]';
%     * Not linearly separable: N = 4; X = (0:3)'; Y = [1 0 1 0]';
%   . A larger step size to speed up the convergence.
%   . Make AX above wide enough to see the whole function landscape. Look at
%     the contours and find the area where the minimiser is.

