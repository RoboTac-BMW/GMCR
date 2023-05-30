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

function [R]  = rot_svd(X, Y)
% Find rotation from a a set of correspondences using SVD method
% The method is described in http://igl.ethz.ch/projects/ARAP/svd_rot.pdf


%X=[M(i,:)', M(j,:)'];
%Y=[B(i,:)', B(j,:)'];

% Assemble the correlation matrix H = X * Y'
%H=X*Y';
H=X'*Y;
% H = USV'
[U,~,V]=svd(H);

% Compute R = V * U'
if det(U) * det(V) < 0
    V(:,3) = -V(:,3);
end
R=V*U';



end