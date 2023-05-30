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

function [res] = pcresolution(X)
% Estimate the resolution of the point cloud X

%Sample 5% points

n = floor(X.Count/20); % sample size
D = zeros(10*n,1);
rng(1)
for i=1:10 %repeat 10 times
    sampleIdx = randsample(length(X.Location),n);
    Xsample = X.Location(sampleIdx,:);
    
    %Remove sampled points from X
    
    kdt = KDTreeSearcher(X.Location(setdiff(1:X.Count, sampleIdx),:));
    [~,D_aux]=knnsearch(kdt, Xsample);
    D(n*(i-1)+1:n*(i-1)+n) = D_aux;
end
res = double(median(D));
return