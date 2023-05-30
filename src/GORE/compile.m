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

% gore4dof     
mex  CXXFLAGS='$CXXFLAGS -Wall -std=c++11' ...
      -lmwblas ...
      -Iinclude ...
      src/gore4.cpp ...
      src/imp_gore4.cpp 
     
% gore6
mex CXXFLAGS='$CXXFLAGS -Wall -std=c++11' ...
    -lmwblas ...
    -Iinclude ...
    src/gore6.cpp ...
    src/imp_gore6.cpp ...
    src/imp_gore3_l2.cpp ...
    src/imp_rot3_matches.cpp ...
    src/geometry.cpp 