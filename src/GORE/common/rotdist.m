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

function [ theta ] = rotdist( S, R)
%Angular distance between two rotation matrices. 

n = norm(S-R,'fro');
theta = 2*asin(n/(2*sqrt(2)));
end

