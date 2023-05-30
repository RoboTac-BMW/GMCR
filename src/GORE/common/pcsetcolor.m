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

function pcsetcolor( cloud, color )

if color=='r'
    cloud.Color = uint8(repmat([255 0 0],cloud.Count,1));
elseif color=='m'
    cloud.Color = uint8(repmat([255 0 255],cloud.Count,1));

elseif color=='b'
    cloud.Color = uint8(repmat([0 0 255],cloud.Count,1));    
elseif color=='grey'
    cloud.Color = uint8(repmat([180 180 180],cloud.Count,1));    
end

end

