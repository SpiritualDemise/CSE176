% This code demonstrates Laplacian K-modes (using the homotopy training) and
% its out-of-sample mapping on the 2-moons dataset.

rng(0);

load 2moons.mat X N AX

% In this example, we set K to 2 (the known number of clusters) and we set the
% kernel bandwidth to 0.1 (which gives a reasonably good density estimate). To
% illustrate the homotopy algorithm, we decrease the bandwidth s from large
% (s=5) to its target (s=0.2) in a geometric sequence of 10 values. The
% trade-off parameter lambda=1 is fixed.

K = 2;						% Number of clusters
S = 10.^linspace(log10(5),log10(0.1),10);	% Bandwidth values
lambda = 1;					% Trade-off parameter

% We build a 5-nn graph to compute the graph Laplacian
knn = 5; W = gaussaff(X,{'k',knn},0.1); d = sum(W,2); Laplacian = diag(d)-W;

% Use K-means for initialization
[C,label] = kmeans(X,K);
Z = zeros(N,K); Z(sub2ind([N,K],(1:N)',label)) = 1;

% Run Laplacian K-modes for decreasing values of the bandwidth s (homotopy)
CC = {C}; ZZ = {Z};
for its=1:length(S)
  [C,Z,E] = lapkmodes(X,K,S(its),lambda,Laplacian,[],[],C,Z);
  CC = [CC,{C}]; ZZ = [ZZ, {Z}];
end
S = [Inf S];


% Plot results: dataset, contours, location of the modes and soft assignments
% (all as a function of the bandwidth s during the homotopy).
figure(1); clf; plot(X(:,1),X(:,2),'bo'); daspect([1 1 1]); axis(AX);
figure(2); COLORSX = {'r+','b+','g+'};
for its=1:length(CC)
  C = CC{its}; Z = ZZ{its}; [P,IDX] = max(Z,[],2);
  
  clf; hold on;
  % Plot hard clustering and centroids.
  for k=1:K
    idx = find(IDX==k); plot(X(idx,1),X(idx,2),COLORSX{k},'MarkerSize',8);
  end
  plot(C(:,1),C(:,2),'ko','MarkerSize',16,'MarkerFaceColor','k');
  
  % Plot path
  plot(CC{1}(:,1),CC{1}(:,2),'k*','MarkerSize',16,'MarkerFaceColor','k');
  for i=2:its
    p1 = [CC{i-1}(:,1), CC{i}(:,1), NaN(K,1)]'; p1 = p1(:);
    p2 = [CC{i-1}(:,2), CC{i}(:,2), NaN(K,1)]'; p2 = p2(:);
    plot(p1,p2,'k','LineWidth',2);
  end
  
  % Plot normalized KDE for each cluster.
  if its>1
    for j=1:K
      t1 = linspace(AX(1),AX(2),50); t2 = linspace(AX(3),AX(4),50);
      [Y1,Y2] = meshgrid(t1,t2); Y = [Y1(:),Y2(:)];
      pdf = exp(-sqdist(Y,X)/(2*S(its)^2))*Z(:,j)/sum(Z(:,j))/...
            (sqrt(2*pi)*S(its))^2;
      contour(Y1,Y2,reshape(pdf,50,50),10,COLORSX{j}(1));
    end
  end
  daspect([1 1 1]); axis(AX); box on; title(['\sigma=',num2str(S(its))]);
  pause(0.5)
end

%% Generate out-of-sample data.
XRES = 100; YRES = 66;
xx = linspace(AX(1),AX(2),XRES); yy = linspace(AX(3),AX(4),YRES);
[aa,bb] = meshgrid(xx,yy); X1 = [aa(:),bb(:)];

%% Set the lambda and s used in out-of-sample mapping.
lambda=1; s = 0.1;

%% Compute solution and visualize.
Z1 = lapkmodes_map(X1,C,Z,X,s,lambda,knn);

COLORS = [1 0 0;0 0 1];
C3 = Z1*COLORS(1:K,:);
I3 = zeros(YRES,XRES,3);
I3(:,:,1) = reshape(C3(:,1),YRES,XRES);
I3(:,:,2) = reshape(C3(:,2),YRES,XRES);
I3(:,:,3) = reshape(C3(:,3),YRES,XRES);
figure(33); clf; hold on;
image(xx,yy,I3); plot(X(:,1),X(:,2),'y+');
for k=1:K
  plot(C(k,1),C(k,2),'k>','MarkerSize',20,'MarkerFaceColor','k');
end
set(gca,'DataAspectRatio',[1 1 1]); axis(AX);

%% Plot everything at each homotopy iteration.
figure(99);
for its=1:length(CC)
  C = CC{its};
  Z = ZZ{its};
  s = S(its);
  
  % Clustering, density.
  [confidence,IDX] = max(Z,[],2);
  set(0,'CurrentFigure',99); clf;
  set(gcf,'Position',[200,200,1000,450],'PaperPositionMode','auto');
  axes('Position',[0.04 0.02 0.45 0.9]); hold on;
  for k=1:K
    idx = find(IDX==k);
    plot(X(idx,1),X(idx,2),COLORSX{k},'MarkerSize',8);
  end
  if ~isinf(S)
    for j=1:K   % Plot normalized KDE for each cluster.
      t1 = linspace(AX(1),AX(2),50); t2 = linspace(AX(3),AX(4),50);
      [Y1,Y2] = meshgrid(t1,t2); Y = [Y1(:),Y2(:)];
      pdf = exp(-sqdist(Y,X)/(2*s^2))*Z(:,j)/sum(Z(:,j))/...
            (sqrt(2*pi)*s)^2;
      contour(Y1,Y2,reshape(pdf,50,50),10,COLORSX{j}(1));
    end
  end
  for k=1:K
    plot(C(k,1),C(k,2),'k>','MarkerSize',16,'MarkerFaceColor','k');
  end
  set(gca,'DataAspectRatio',[1 1 1]); axis(AX); box on;
  
  % Out-of-sample points.
  Z1 = lapkmodes_map(X1,C,Z,X,s,lambda,knn);

  C3 = Z1*COLORS(1:K,:);
  I3 = zeros(YRES,XRES,3);
  I3(:,:,1) = reshape(C3(:,1),YRES,XRES);
  I3(:,:,2) = reshape(C3(:,2),YRES,XRES);
  I3(:,:,3) = reshape(C3(:,3),YRES,XRES);
  
  axes('Position',[0.53 0.02 0.45 0.9]); hold on;
  image(xx,yy,I3); plot(X(:,1),X(:,2),'y+');
  for k=1:K
    plot(C(k,1),C(k,2),'k>','MarkerSize',16,'MarkerFaceColor','k');
  end
  set(gca,'DataAspectRatio',[1 1 1]); axis(AX);
  
  % Title.
  axes('Position',[0.35 0.92 0.5 0.05]); set(gca,'FontSize',32);
  if its==1 s=Inf; end
  text(0,0,['K=2,   \lambda=',num2str(lambda),',   \sigma=',num2str(s)],...
       'fontsize',28);
  axis tight; axis off;
  pause(0.5);
  
end

