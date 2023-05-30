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

function [H, D] = matchkps(hist1, hist2, Xkps, Ykps, desiredMatches)
% Match keypoints using pfh histograms

%assert(desiredMatches>100, 'too low desired matches? probably using prev. version???');

% normalRad = .5* searchRad; % normal rad must be < searchRad
% 
% hist1 = pfh(X, Xkps, searchRad, normalRad);
% hist2 = pfh(Y, Ykps, searchRad, normalRad);

D = zeros(desiredMatches,1); %allocate
H = zeros(desiredMatches,2, 'uint32');
k=1; % Count
maxD=0;
maxDIdx=1;
info_count=0; %to show progress
for i=1:size(Xkps,1)
%     if (mod(i-1,200)==0)
%         if i>0
%         fprintf(1, repmat('\b',1,info_count));
%         end
%         info_count = fprintf('completed %.00f%%\n', 100*i/size(Xkps,1));
%     end
    
    hi = hist1(i,:);
    for j=1:1:size(Ykps,1)
        dvec=hi-hist2(j,:);
        d = norm(dvec);
        
        if k<=desiredMatches
            if d>maxD
                maxD=d;
                maxDIdx=k;
            end
            D(k)=d;
            H(k,:)=[i,j];
            k = k+1;
        elseif d<maxD
            D(maxDIdx)=d;
            H(maxDIdx,:)=[i,j];
            
            % update maxD
            [~,maxDIdx]=max(D);
            maxD = D(maxDIdx);
        end
    end
end

D=D(1:k-1,:);
H=H(1:k-1,:);

%[D,idx] = sort(D);
%H = H(idx,:);

end
