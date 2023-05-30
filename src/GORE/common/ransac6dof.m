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

function [ opt_val, T, iter ] = ransac6dof(X, Y, th, p)
% Ransac for rotation over angular error.
% point correspondences are such that (X(i,:), Y(i,:))
% Find rotation from a minimal set of correspondences using SVD method
% The method is described in http://igl.ethz.ch/projects/ARAP/svd_rot.pdf

numOfPts = size(X,1);

opt_val = 0;
opt_R = eye(3);
opt_tx = zeros(3,1);
opt_ty = zeros(3,1);

%inliers ratio
iter=0;
maxIters=inf;
while iter<maxIters
    % Sample 3 points
    sample_idx=randsample(numOfPts,3);
    X_sample = X(sample_idx,:)';
    Y_sample = Y(sample_idx,:)';
    
    % Centre
    tx = mean(X_sample, 2);
    ty = mean(Y_sample, 2);
    
    X_sample = X_sample - repmat(tx,1,3);
    Y_sample = Y_sample - repmat(ty,1,3);
    
    % Assemble the correlation matrix H = X * Y'
    H=X_sample*Y_sample';
    
    % H = USV'
    [U,~,V]=svd(H);
    
    % Compute R = V * U'
    if det(U)*det(V) < 0
        V(:,3) = -V(:,3);
    end
    R=V*U';
    
    % Evaluate rotation
    Xc = X - repmat(tx',numOfPts,1);
    Yc = Y - repmat(ty',numOfPts,1);
    
    X2 = (R*Xc')';
    res = X2-Yc;
    res = res(:,1).^2 + res(:,2).^2 + res(:,3).^2;
    val = sum(res<=th*th);
    
    % Update solution and number of iterations
    if val>opt_val
        opt_val = val;
        opt_R = R;
        opt_tx = tx;
        opt_ty = ty;
        w = opt_val/numOfPts;
        maxIters = ceil(log(1-p)/log(1-w^3));
    end
    iter=iter+1;
end


% Obtain opt_T from inliers
if opt_val>2
    
    Xc = X - repmat(opt_tx',numOfPts,1);
    Yc = Y - repmat(opt_ty',numOfPts,1);
    
    res = Xc*opt_R'-Yc;
    res = res(:,1).^2 + res(:,2).^2 + res(:,3).^2;
    inlrs_idx = res<=th*th;
    
    X_sample = X(inlrs_idx,:);
    Y_sample = Y(inlrs_idx,:);
    
    tx = mean(X_sample)';
    ty = mean(Y_sample)';
    
    Xc = X - repmat(tx',numOfPts,1);
    Yc = Y - repmat(ty',numOfPts,1);
    
    res = Xc*opt_R'-Yc;
    res = res(:,1).^2 + res(:,2).^2 + res(:,3).^2;
    inlrs_idx = res<=th*th;
    
    val = sum(inlrs_idx);
    
    if (val>opt_val)
        opt_val = val;
        opt_tx = tx;
        opt_ty = ty;
    end
    
    % Compute roation
    X_sample = Xc(inlrs_idx,:);
    Y_sample = Yc(inlrs_idx,:);
    R = rot_svd(X_sample, Y_sample);
    
    % evaluate R,tx,ty
    res = Xc*R'-Yc;
    
    res = res(:,1).^2 + res(:,2).^2 + res(:,3).^2;
    val = sum(res<=th*th);
    
    if (val>opt_val)
        opt_val = val;
        opt_R = R;
        opt_tx = tx;
        opt_ty = ty;
    end
    
end

Cx=[eye(3) -opt_tx; 0 0 0  1];
Cy_inv =[eye(3) opt_ty; 0 0 0  1];
T = Cy_inv * [opt_R [0 0 0]'; 0 0 0  1] * Cx;

end

