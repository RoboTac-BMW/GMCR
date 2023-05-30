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

#ifndef ROT3_MATCHES_EVALUATOR_
#define ROT3_MATCHES_EVALUATOR_

#include "reg_common.h"
#include "data_indexation.h"
#include "state.h"

namespace reg {
namespace search { 

template<class SSR>
class Rot3MatchesEvaluator : public DataIndexation<SSR>
{
public:
    Rot3MatchesEvaluator( const Matrix3X &X, const Matrix3X &Y, double th );
    Rot3MatchesEvaluator();

    int sweep();
    int sweep(int *matchList, int matchListSize, std::vector<bool> &matches);

    int size() const;

    int evalUpperBound(SSR ssr, int lwbnd) const;
    int evalUpperBound(SSR ssr, int lwbnd,
                       int *matchList, int matchListSize,
                       std::vector<bool> &matches) const;

    int evalLowerBound(SSR ssr) const;
    int evalLowerBound(SSR ssr, int *matchList, int matchListSize) const;
    int evalLowerBound(SSR ssr, int *matchList, int matchListSize,
                       std::vector<bool> &matches) const;


private:

    const Matrix3X &M_in;
    const Matrix3X &B_in;
    const double th;

    Matrix3X M;
    Matrix3X B;

    int _size;
};

    
} // End namespace sarch
} // End namespace reg

#include "rot3_matches_evaluator.hpp"


#endif

