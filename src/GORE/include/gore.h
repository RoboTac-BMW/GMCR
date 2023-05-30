/**
*                    GUARANTEED OUTLIER REMOVAL 
*         FOR POINT CLOUD REGISTRATION WITH CORRESPONDENCES
*
*
*
* Copyright (C) 2016 Alvaro PARRA BUSTOS (aparra at cs.adelaide.edu.au)
* School of Computer Science, The University of Adelaide, Australia
* The Australian Center for Visual Technologies
* http://cs.adelaide.edu.au/~aparra
* 
* The source code, binaries and demo is distributed for academic use only.
* For any other use, including any commercial use, contact the authors.
*/

#ifndef REG_GORE_
#define REG_GORE_

#include "reg_common.h"
//#include "state.h"

namespace reg {
    namespace gore {
        
        // Euclidean version
        int gore3_l2(size_t *H, const double *X, const double *Y, int numOfPts,
                     double th, int lwbnd, bool repFlag,
                     double *R, int &outLwbnd);
        
        
        int gore4(size_t *H, const double *X, const double *Y, int numOfPts,
                  double th, int lwbnd, bool repFlag,
                  Transform3 &T, int &outLwbnd);
        
        
        int gore6(size_t *H, const double *X, const double *Y, int numOfPts,
                  double th, int lwbnd, bool repFlag,
                  Transform3 &T, int &outLwbnd);
        
    } // End namespace sarch
} // End namespace reg

#endif

