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

A demostration of gore for 6 DoF is given in demo.m

To compile gore execute
  compile


---------------------------------------------------------------------------
-- DEMO
---------------------------------------------------------------------------

The number of used point correspondences is set in N (line 21).

To chose another dataset change variable dataset (line 22) which can accept
values in {'stanford', 'mian', 'vaihingen', 'mining'}.

To chose another model instance change variable modelIdx (line 23). There 
are 4 models for stanford and mian and 2 for vaihingen and mining.
