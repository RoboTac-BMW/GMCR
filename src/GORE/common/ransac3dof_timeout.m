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

function [ opt_val, opt_R, iter, maxIters ] = ransac3dof_timeout(X, Y, th, p, s)
% Ransac for rotation over angular error.
% point correspondences are such that (X(i,:), Y(i,:))
% Find rotation from a minimal set of correspondences using SVD method
% The method is described in http://igl.ethz.ch/projects/ARAP/svd_rot.pdf
% s: timeout. If <0 run all iterations

numOfPts = size(X,1);

opt_val = 0;
opt_R = eye(3);

%inliers ratio
iter=0;
maxIters=inf;
tic;
while (iter < maxIters) && (toc<s || s<0 )
    
%     if mod(iter,10000)==0
%     fprintf('completed %f %%...\n',100*iter/maxIters);
%     end
    % Sample 2 points
    sample_idx = randsample(numOfPts,2);
    X_sample = X(sample_idx,:)';
    Y_sample = Y(sample_idx,:)';
    
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
    X2 = X*R';
    Idx = evalcsm(X2,Y,th);
    val = sum(Idx);
    
    % Update solution and number of iterations
    if val>opt_val
        opt_val = val;
        opt_R = R;
        w = opt_val/numOfPts;
        maxIters = ceil(log(1-p)/log(1-w^2));
    end
    iter=iter+1; 
end


% Obtain opt_R from inliers
if opt_val>2
    X2 = X*(opt_R');
    Idx = evalcsm(X2,Y,th);
    val = sum(Idx);
    if (val>opt_val)
        opt_val = val;
        opt_R=rot_svd(X(Idx,:), Y(Idx,:));
    end
end

end
