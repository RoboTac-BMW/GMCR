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

#ifndef REG_SEARCH_
#define REG_SEARCH_

#include "reg_common.h"
#include "data_indexation.h"

namespace reg {
    namespace search {
        
        /**
         * @brief Find optimal quality of a transformation
         * @param psi Indexation of point sets.
         * @param knownSolutionQual Known solution quality.
         * @param gap BnB stop gap.
         * @param guessAndResult Guess search region and final region such that
         *        upbnd-lwbnd<=gap
         * @return Quality of the central transform of the optimal region.
         */
        template <class SSR, unsigned int BRANCHING_FACTOR>
        int bnb_search_queue(const DataIndexation<SSR> &psi,
                             int lwbnd, int gap, SSR &guessAndResult);
        
        
        template <class SSR, unsigned int BRANCHING_FACTOR>
        int bnb_search_table(const DataIndexation<SSR> &dsi,
                             int lwbnd, int gap, int buckets,
                             SSR &guessAndResult);
        
        template <class SSR, unsigned int BRANCHING_FACTOR>
        int searchTableDF(const DataIndexation<SSR> &dsi,
                          int lwbnd, int gap, int buckets,
                          SSR &guessAndResult);
        
        
        template <class SSR, unsigned int BRANCHING_FACTOR>
        int bnb_search_ml_table(const DataIndexation<SSR> &dsi,
                                int lwbnd, int gap, int buckets,
                                SSR &guessAndResult);
        
        
    } // End namespace sarch
} // End namespace reg

#include "search.hpp"

#endif

