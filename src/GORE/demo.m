%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    GUARANTEED OUTLIER REMOVAL 
%         FOR POINT CLOUD REGISTRATION WITH CORRESPONDENCES
%
%
%
% Copyright (C) 2016 Alvaro PARRA BUSTOS (aparra at cs.adelaide.edu.au)
% School of Computer Science, The University of Adelaide, Australia
% The Australian Center for Visual Technologies
% http://cs.adelaide.edu.au/~aparra
% 
% The source code, binaries and demo is distributed for academic use only.
% For any other use, including any commercial use, contact the authors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
addpath data common
%compile

%% Parametres
N = 1000;
dataset = 'stanford'; %stanford (4) mian (4) vaihingen (2) mining (2)
modelIdx = 1;

load(sprintf('data_%s_%d', dataset, modelIdx))

th = mean([pcresolution(conf.v1) pcresolution(conf.v2)]);


[H,~] = matchkps(conf.hist1, conf.hist2, ...
    conf.kps1.Location, conf.kps2.Location, N);

% Initialise
% Generate random transforms
rng('shuffle')
T_init = affine3d([randrot [0 0 0]'; 100*(rand(1,3)*2-1) 1]);

kps1_init = pctransform(conf.kps1,T_init);

% Ground truth
G = T_init.T \ conf.GT1 / conf.GT2;



%% GORE
fprintf('GORE...\n');
rep_flag=1;

X = pointCloud(kps1_init.Location(H(:,1),:));
Y = pointCloud(conf.kps2.Location(H(:,2),:));

g=struct();

Xin=double(X.Location');
Yin=double(Y.Location');
% warm-up
[~,~,~]=gore6(Xin,Yin,th,0,0);
tic
[g.H, g.T, g.lwbnd] = gore6(Xin,Yin,th,0, rep_flag);
g.time=toc();
g.T=g.T';
g.angErr=rotdist(G(1:3,1:3), g.T(1:3,1:3));
g.trErr=norm(G(1:3,1:3)*G(4,1:3)'-g.T(1:3,1:3)*g.T(4,1:3)');
fprintf('|H''| = %d\n', length(g.H));


%% GORE+RANSAC
fprintf('GORE+RANSAC...\n');

RANSAC_TIMEOUT=5*60*60;

H2 = H(g.H,:);
X = pointCloud(kps1_init.Location(H2(:,1),:));
Y = pointCloud(conf.kps2.Location(H2(:,2),:));

gore_rsc=struct();
tic
[gore_rsc.val, gore_rsc.T, gore_rsc.iter, gore_rsc.totalIter] = ...
    ransac6dof_timeout(X.Location, Y.Location, th, .99, RANSAC_TIMEOUT);
gore_rsc.time=toc();
gore_rsc.T = gore_rsc.T'; % Use left mult.
gore_rsc.angErr = rotdist(G(1:3,1:3), gore_rsc.T(1:3,1:3));
gore_rsc.trErr  = norm(G(1:3,1:3)*G(4,1:3)'-gore_rsc.T(1:3,1:3)*gore_rsc.T(4,1:3)');


%% Plot
v1 = conf.v1;
v2 = conf.v2;

kps1 = conf.kps1;
kps2 = conf.kps2;

v1_init = pctransform(v1,T_init);
kps1_init = pctransform(kps1,T_init);

if strcmp(dataset,'stanford')
    plotT=affine3d([[1 0 0; 0 0 1; 0 -1 0] [0 0 0]'; 0 0 0  1]);
elseif strcmp(dataset,'mian')
    plotT=affine3d([[1 0 0; 0 0 -1; 0 1 0] [0 0 0]'; 0 0 0  1]);
else
    plotT=affine3d(eye(4));
end

% Input
pcsetcolor(v1_init,'m');
pcsetcolor(v2,'b');

figure('Color','w')
subplot(2,2,1)

pcshow(pctransform(v1_init,plotT))
hold on
pcshow(pctransform(v2,plotT))

Xplot=pctransform(pointCloud(kps1_init.Location(H(:,1),:)),plotT);
Yplot=pctransform(pointCloud(kps2.Location(H(:,2),:)),plotT);

X=pointCloud(kps1_init.Location(H(:,1),:));
Y=pointCloud(kps2.Location(H(:,2),:));

for i=1:size(H,1)
    xi = Xplot.Location(i,:);
    yi = Yplot.Location(i,:);
    plot3([xi(1) yi(1)], [xi(2) yi(2)], [xi(3) yi(3)], '-r')
end
title('Input')
axis off
drawnow


% Plot correspondences after removing outliers
H2 = H(g.H,:);

X2 = pointCloud(kps1_init.Location(H2(:,1),:));
Y2 = pointCloud(kps2.Location(H2(:,2),:));

subplot(2,2,2);
pcshow(pctransform(v1_init,plotT))

hold on
pcshow(pctransform(v2,plotT))
Xplot = pctransform(X2,plotT);
Yplot = pctransform(Y2,plotT);
for i=1:X2.Count
    x = Xplot.Location(i,:);
    y = Yplot.Location(i,:);
    plot3([x(1) y(1)],[x(2) y(2)], [x(3) y(3)], '-r')
end
title('GORE''S output (H'')')
axis off
drawnow

% Plot transformation
subplot(2,2,3);
v1_reg = pctransform(v1_init,affine3d(g.T));
pcshow(pctransform(v1_reg, plotT))
hold on
pcshow(pctransform(v2, plotT))
title('GORE''s alignment')
axis off
drawnow

% Plot RANSAC
subplot(2,2,4)
v1_rsc = pctransform(v1_init,affine3d(gore_rsc.T));
pcshow(pctransform(v1_rsc,plotT))
hold on
pcshow(pctransform(v2,plotT))
title('RANSAC on H''')
axis off
drawnow

%%  Info
fprintf('\n----------------------------------------------------------------\n')
fprintf('Data\n')
fprintf('  Number of points      : |X| = %d  |Y| = %d\n', size(v1.Location,1), size(v2.Location,1));
fprintf('  N (number of matches) : %d\n\n',size(H,1));

fprintf('GORE\n')
fprintf('  time (s)                : %.2f\n', g.time)
fprintf('  N'' (remaining matches)  : %d\n', length(g.H))
fprintf('  objective value         : %d\n',   g.lwbnd);
fprintf('  ang. error (degrees)    : %.4f\n', g.angErr*180/pi);
fprintf('  translation error       : %.4f\n\n', g.trErr);

fprintf('RANSAC on H''\n')
fprintf('  time (s)             : %.3f\n', gore_rsc.time)
fprintf('  objective value      : %d\n',   gore_rsc.val);
fprintf('  ang. error (degrees) : %.4f\n', gore_rsc.angErr*180/pi);
fprintf('  translation error    : %.4f\n', gore_rsc.trErr);