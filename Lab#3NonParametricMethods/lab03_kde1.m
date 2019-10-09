% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Demonstrate nonparametric methods:
% kernel density estimate (KDE) for density estimation, classification and
% regression, on 1D toy datasets (for Gaussian kernels, using the GM tools).

% ---------- 1D dataset: made of 3 pieces ----------

% First try a simpler dataset by using only the "x2" piece (linear fcn).
f = @(x) (x<2).*(0) + (x>=2).*(x<3).*(2*(x-2)+1) + (x>=3).*(2*(x-4).^2); % regr
fc = @(x) (x<2).*(1) + (x>=2).*(x<3).*(2) + (x>=3).*(1);	% class label
fd = @(x) (x<2).*(1/6*exp(-(x/0.3).^2/2)/sqrt(2*pi)/0.3) + ...	% density
                  (x>=2).*(x<3).*(1/2*1) + (x>=3).*(1/3*1/2);
rng(1778);
x1 = randn(10,1)*0.3; x2 = rand(30,1)+2; x3 = 2*rand(20,1)+3;	% pieces
X = [x1;x2;x3]; N=length(X); C=fc(X); K=max(C); Y = f(X)+randn(N,1)/10;
x = linspace(-1,5,500)'; fx = f(x); fdx = fd(x);	% true fcn and density
ax = [-2 6 -1 4];


% ---------- Density estimation with a histogram ----------

% We vary the histogram's origin x0 and bin width h

figure(1); set(gcf,'Name','1D dataset density estimation with histogram');
h = 0.5;
for x0=linspace(-1,-0.5,10)		% vary origin x0, fixed bin width h
  set(0,'CurrentFigure',1); clf; [H,Hc] = hist(X,x0+h/2:h:5.5);
  hh1 = bar(Hc,5*H/sum(H)/h,1,'FaceColor','r','EdgeColor','r'); hold on;
  hh2 = plot(X,0*X,'bo',x,5*fdx,'g-',...
       ax(1)+[0 h],-0.8*[1 1],'k>-',[x0 x0],ax([3 4]),'k:');
  text(ax(1)+h,-0.8,'  h'); text(x0,ax(4)+0.1,'x_0');
  hold off; axis(ax); daspect([1 1 1]); box on;
  xlabel('x'); ylabel('p(x)'); legend([hh1 hh2(2)],'hist','true');
  title(['Histogram with x_0 = ' num2str(x0) ', h = ' num2str(h)]);
  pause
end

figure(2); set(gcf,'Name','1D dataset density estimation with histogram');
x0 = -1;
for h=10.^linspace(-2,0.5,20)		% vary bin width h, fixed origin x0
  set(0,'CurrentFigure',2); clf; [H,Hc] = hist(X,x0+h/2:h:5.5);
  hh1 = bar(Hc,5*H/sum(H)/h,1,'FaceColor','r','EdgeColor','r'); hold on;
  hh2 = plot(X,0*X,'bo',x,5*fdx,'g-',...
       ax(1)+[0 h],-0.8*[1 1],'k>-',[x0 x0],ax([3 4]),'k:');
  text(ax(1)+h,-0.8,'  h'); text(x0,ax(4)+0.1,'x_0');
  hold off; axis(ax); daspect([1 1 1]); box on;
  xlabel('x'); ylabel('p(x)'); legend([hh1 hh2(2)],'hist','true');
  title(['Histogram with x0 = ' num2str(x0) ', h = ' num2str(h)]);
  pause
end


% ---------- Density estimation with a Gaussian KDE ----------

% We vary the KDE's bandwidth h
% Construct a KDE; we reuse the GM struct:
kde.p = ones(N,1)/N; kde.c = X; kde.type = 'i';

figure(3); set(gcf,'Name','1D dataset density estimation with Gaussian KDE');
for h=10.^linspace(-2,0.5,20)			% vary bandwidth h
  kde.S = h^2;
  kde_p = GMpdf(x,kde);						% KDE pdf
  set(0,'CurrentFigure',3); clf; hold on;
  plot(X,0*X,'bo',ax(1)+[0 h],-0.8*[1 1],'k>-'); text(ax(1)+h,-0.8,'  h');
  hh = plot(x,5*kde_p,'r-',x,5*fdx,'g-');
  hold off; axis(ax); daspect([1 1 1]); box on; legend(hh,'KDE','true');
  title(['Gaussian KDE with h = ' num2str(h)]); xlabel('x'); ylabel('p(x)');
  pause
end


% ---------- Classification with a Gaussian KDE ----------

% We vary the KDE's bandwidth h
% Construct a KDE for each class; we reuse the GM struct:
for k=1:K
  kdeC{k}.i = find(C==k);			% indices of points in class k
  kdeC{k}.N = length(kdeC{k}.i);		% number of points in class k
  kdeC{k}.type = 'i';				% isotropic, homoscedastic GM
  kdeC{k}.p = ones(kdeC{k}.N,1)/kdeC{k}.N;	% proportions = 1/Nk
  kdeC{k}.c = X(kdeC{k}.i);			% centers = points in class k
end

% Colors for points
COL = [1 0 0;0 0 1;0 1 0;0 0 0;1 1 0;1 0 1;0 1 1;0.75 0.4 0;1 0.6 0.6;1 0.8 0];
figure(4); set(gcf,'Name','1D dataset classification with Gaussian KDE');
for h=10.^linspace(-2,0.5,20)			% vary bandwidth h
  set(0,'CurrentFigure',4); clf; colormap(COL(1:K,:)); hold on;
  plot(ax(1)+[0 h],-0.8*[1 1],'k>-'); text(ax(1)+h,-0.8,'  h');
  kdeC_p = zeros(length(x),K); kdeC_p1 = zeros(N,K);
  for k=1:K
    kdeC{k}.S = h^2;
    kdeC_p1(:,k) = GMpdf(X,kdeC{k})*kdeC{k}.N/N;		% KDE postprobs
    kdeC_p(:,k) = GMpdf(x,kdeC{k})*kdeC{k}.N/N;
    scatter(X(kdeC{k}.i),-0.5*ones(kdeC{k}.N,1),24,k*ones(kdeC{k}.N,1));
    plot(x,5*kdeC_p(:,k),'-','Color',COL(k,:));
  end
  [~,CC] = max(kdeC_p1,[],2); scatter(X,zeros(N,1),24,CC);
  hold off; axis(ax); daspect([1 1 1]); box on;
  title(['Gaussian KDE per class with h = ' num2str(h)]);
  xlabel('x'); ylabel('p(k|x)');
  text(min(X),-0.05,'PREDICTED:  ','HorizontalAlignment','right');
  text(min(X),-0.55,'TRUE:  ','HorizontalAlignment','right');
  pause
end


% ---------- Regression with a Gaussian KDE ----------

% We vary the KDE's bandwidth h
% Construct a KDE; we reuse the GM struct:
kdeR.p = ones(N,1)/N; kdeR.c = [X Y]; kdeR.type = 'i';

figure(5); set(gcf,'Name','1D dataset regression with Gaussian KDE');
for h=10.^linspace(-2,0.5,20)			% vary bandwidth h
  kdeR.S = h^2;
  kdeR_f = GMmeanrec([x x],repmat([1 0],length(x),1),kdeR);	% KDE regress.
  set(0,'CurrentFigure',5); clf; hold on;
  plot(X,Y,'b+',ax(1)+[0 h],-0.8*[1 1],'k>-'); text(ax(1)+h,-0.8,'  h');
  hh = plot(x,kdeR_f(:,2),'r-',x,fx,'g-');
  hold off; axis(ax); daspect([1 1 1]); box on; legend(hh,'KDE','true');
  title(['Gaussian KDE with h = ' num2str(h)]); xlabel('x'); ylabel('y');
  pause
end


% Suggestions of things to try:
%
% - In general:
%   . Vary the bandwidth.
%   . Use other kernel functions beyond the Gaussian.
%   . Try other 1D datasets having different distribution, noise levels, etc.
% - Classification:
%   . Different numbers of classes.
%   . Classes with different shapes and overlap.
% - 2D datasets:
%   . Plot posterior probabilities p(k|x) as contours and as a color image.
%   . Plot classification result argmax_k{p(k|x)} as a color image.
%   Use the plots in lab02.m as a guideline.

