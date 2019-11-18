% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Demonstrate (stochastic) gradient descent in 1D datasets with:
% - Linear regression: f(x) = W.x + w
%   trained with least-squares error.

% Dataset for 1D regression: we pick data having nonzero mean and nonunit
% stdev so the objective function is an elongated quadratic form.
rng(1778);
N = 100; Nv = 500; D = 1; E = 1; a = -0.6; b = 0.4; s = 0.1;
X = 3*rand(N,D); Y = a*X + b + s*randn(N,E);		% Training set
Xv = 3*rand(Nv,D); Yv = a*Xv + b + s*randn(Nv,E);	% Validation set
ax = [0 3 -1.7 0.7];					% Range: y-x plot
AX = {[-1.2 0.2;-0.2 1],[-0.67 -0.57;0.4 0.48]};	% Ranges: contour plots

% Linear regression, exactly by solving the normal equations
f = linfexact(X,Y);
e = Y - linf(X,f); ef = e(:)'*e(:)/2;			% Optimal error

% Initial weights for both GD and SGD
g0.W = randn(E,D)/1e3; g0.w = randn(E,1)/1e3; g0.type = 'linf';
maxit = 100;						% Max # iterations

% Linear regression, iteratively with gradient descent
eta = 4.5e-3;						% Step size
[GDg,GDe] = linfgd(X,Y,Xv,Yv,eta,maxit,g0);

% Linear regression, iteratively with stochastic gradient descent
Eta = 5e-2;						% Step size
B = 1;							% Minibatch size
[SGDg,SGDee] = linfsgd(X,Y,Xv,Yv,Eta,maxit,B,g0);


% ---------------------------------- Plots ----------------------------------

% Put iterates in an array for plotting
GDw = zeros(D+1,maxit+1); for i=1:maxit+1, GDw(:,i) = [GDg{i}.W GDg{i}.w]'; end
SGDww = zeros(D+1,ceil(N/B),maxit+1);			% Minibatch iterates
SGDww(:,:,1) = NaN; SGDww(:,1,1) = [SGDg{1,1}.W SGDg{1,1}.w]';
for i=1:maxit+1
  for n=1:ceil(N/B), SGDww(:,n,i+1) = [SGDg{n,i}.W SGDg{n,i}.w]'; end
end
SGDw = squeeze(SGDww(:,end,:)); SGDe = squeeze(SGDee(:,end,:));	% Iterates

% Objective function values, per data point
GDe = diag([N Nv].^(-1))*GDe;
SGDee(1,:,:) = SGDee(1,:,:)/N; SGDee(2,:,:) = SGDee(2,:,:)/Nv;
SGDe = squeeze(SGDee(:,end,:));

% Dataset and regression line estimates
figure(1); clf; hold on;
set(gcf,'Name','1D dataset; linear regression by least-squares error');
h = plot(X,Y,'bo',Xv,Yv,'g+',...
         ax(1:2),a*ax(1:2)+b,'b-',ax(1:2),f.W*ax(1:2)+f.w,'k-');
h1 = plot(ax(1:2),GDw(1,1)*ax(1:2)+GDw(2,1),'r-','LineWidth',2);
h2 = plot(ax(1:2),SGDw(1,1)*ax(1:2)+SGDw(2,1),'m-','LineWidth',2);
legend([h;h1;h2],'training','validation','true f','optimal f','GD f','SGD f');
axis(ax); daspect([1 1 1]); box on; xlabel('x'); ylabel('y');
title('iteration (epoch): 0');

% Learning curves: error over iterations
figure(2); clf;
set(gcf,'Name','1D dataset; linear regression by least-squares error');
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

% Contour plots: error function and iterates (overall view and zoom in)
for j=1:length(AX)
  figure(1+2*j); clf;
  set(gcf,'Name','1D dataset; linear regression by least-squares error');
  fcontours(@Flsqerr,{g0,X,Y},{},{},{},{},AX{j},50,[50;50]);
  plotseq(GDw','r-'); plotseq(SGDw','m-');
  hold on; xlabel('w'); ylabel('w_0');
  title(['GD/SGD iterations; \eta_{GD} = ' num2str(eta) ', \eta_{SGD} = '...
         num2str(Eta) ', B = ' num2str(B)]);
  plot(f.W,f.w,'k*','MarkerSize',24);
  hh1{j}.a = plotseq(GDw(:,1)','ro'); hh1{j}.b = plotseq(SGDw(:,1)','mo');
  figure(2+2*j); clf;
  set(gcf,'Name','1D dataset; linear regression by least-squares error');
  fcontours(@Flsqerr,{g0,X,Y},{},{},{},{},AX{j},50,[50;50]);
  hold on; xlabel('w'); ylabel('w_0');
  title(['SGD minibatches; \eta_{SGD} = ' num2str(Eta) ', B = ' num2str(B)]);
  plot(f.W,f.w,'k*','MarkerSize',24);
  hh2{j}.a = plotseq(GDw(:,1)','ro'); hh2{j}.b = plotseq(SGDw(:,1)','mo');
end

% Plots over iterates
for i=2:maxit+1
  pause;
  set(0,'CurrentFigure',1); delete(h1); delete(h2);
  h1 = plot(ax(1:2),GDw(1,i)*ax(1:2)+GDw(2,i),'r-','LineWidth',2);
  h2 = plot(ax(1:2),SGDw(1,i)*ax(1:2)+SGDw(2,i),'m-','LineWidth',2);
  title(['iteration (epoch): ' num2str(i-1)]);
  set(0,'CurrentFigure',2); delete(h3); delete(h4);
  subplot(2,1,1); h3 = plot(i-1,GDe(1,i),'ro',i-1,SGDe(1,i),'mo');
  subplot(2,1,2); h4 = plot(i-1,SGDe(1,i),'mo');
  for j=1:length(AX)
    set(0,'CurrentFigure',1+2*j); delete(hh1{j}.a); delete(hh1{j}.b);
    hh1{j}.a = plotseq(GDw(:,i)','ro'); hh1{j}.b = plotseq(SGDw(:,i)','mo');
    set(0,'CurrentFigure',2+2*j); delete(hh2{j}.a); delete(hh2{j}.b);
    hh2{j}.a = plotseq(GDw(:,i)','ro'); hh2{j}.b = plotseq(SGDww(:,:,i)','m-');
  end
  drawnow;
end


% Suggestions of things to try:
% - GD: eta = 1e-4 1e-3 4.5e-3 5e-3 6e-3 7e-3 1e-2.
% - SGD: Eta = 1e-4 1e-3 1e-2 5e-2 1e-1 2e-1 3e-1 5e-1; B = 1 10 50;
%   shuffling or not shuffling.
% - Vary the initial weights W, w.
% - Dataset: vary noise level s, #points N, try X with mean=0 and stdev=1.

