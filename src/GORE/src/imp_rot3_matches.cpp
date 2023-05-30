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

#include "reg_common.h"
#include "search.h"
#include "registration.h"
#include "assert.h"
#include "state_priority_hashtable.h"
#include "rot3_matches_evaluator.h"
#include "rot3_matches_evaluator_l2.h"
#include <iostream>

using namespace reg::search;

// ang distance

int reg::search::rot3_matches(const Matrix3X &X, const Matrix3X &Y, double th,
                              int lwbnd, int gap, AxisAngle &result)
{
    typedef RotationSearchSpaceRegion3DOFS8 SSR;
    
    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
    DataIndexation<SSR> *dsi = new Rot3MatchesEvaluator<SSR>(X, Y, th);
    size_t rempts = dsi->sweep();
    
    std::cout<<"sweep out num of pts = "<< rempts<<std::endl;
    
    mxAssert(rempts>0, "rempts>0");
    
    int buckets = MAX(10, (int)X.cols()/10);
    const int qual = bnb_search_table<SSR,8>(*dsi, lwbnd, gap, buckets,guessAndResult);
    delete dsi;
    result = centre(guessAndResult);
    return qual;
}

//Match list
int reg::search::rot3_matches_ml(const Matrix3X &X, const Matrix3X &Y, double th,
                                 int lwbnd, int gap, AxisAngle &result)
{
    typedef RotationSearchSpaceRegion3DOFS8 SSR;
    
    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
    DataIndexation<SSR> *dsi = new Rot3MatchesEvaluator<SSR>(X, Y, th);
    dsi->sweep();
    int buckets = MAX(10, (int)X.cols()/10);
    const int qual = bnb_search_ml_table<SSR,8>(*dsi, lwbnd, gap, buckets,
                                                guessAndResult);
    delete dsi;
    result = centre(guessAndResult);
    return qual;
}


// l2 distance


int reg::search::rot3_matches_l2(const Matrix3X &X, const Matrix3X &Y, double th,
                                 int lwbnd, int gap, AxisAngle &result)
{
    mxAssert(gap>=0, "incorrect gap");
    mxAssert(lwbnd>=0, "incorrect lower bound");
    
    typedef RotationSearchSpaceRegion3DOFS8 SSR;
    
    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
    DataIndexation<SSR> *dsi = new Rot3MatchesEvaluatorL2<SSR>(X, Y, th);
    dsi->sweep();
    int buckets = MAX(10, (int)X.cols()/10);
    int qual = bnb_search_table<SSR,8> (*dsi,lwbnd,gap,buckets,guessAndResult);
    
    delete dsi;
    result = centre(guessAndResult);

    return qual;
}


int reg::search::rot3_matches_l2_ml(const Matrix3X &X, const Matrix3X &Y, double th,
                                    int lwbnd, int gap, AxisAngle &result)
{
    mxAssert(gap>=0, "incorrect gap");
    mxAssert(lwbnd>=0, "incorrect lower bound");

    typedef RotationSearchSpaceRegion3DOFS8 SSR;
    
    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
    DataIndexation<SSR> *dsi = new Rot3MatchesEvaluatorL2<SSR>(X,Y,th);
    dsi->sweep();
    
    int buckets = MAX(10, (int)X.cols()/10);
    int qual = bnb_search_ml_table<SSR,8>(*dsi,lwbnd,gap,buckets,guessAndResult);
    
    delete dsi;
    result = centre(guessAndResult);
    
    mxAssert(qual<=X.n, "bung in BnB?");
    return qual;
}



//int reg::search::bnb_rsearch_3dof_s8_1kdt_ml(const Matrix &M, const Matrix &B,
//                                             double th, int gap, int known_sol_qual,
//                                             int buckets, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new Rot1KDTDataSetIndexation<SSR>(M, B, th);
//    dsi->sweep();

//    const int qual = bnb_search_ml_table<SSR, 8> (*dsi, known_sol_qual, gap,
//                                                  buckets, guessAndResult);
//    delete dsi;

//    //result = Eigen::AngleAxisd(ssrAngle(guessAndResult), ssrAxis(guessAndResult));
//    result = centre(guessAndResult);
//    return qual;
//}




//int reg::search::bnb_rsearch_3dof_s8_nkdt_ml(const Matrix &M, const Matrix &B,
//                                             double th, int gap, int known_sol_qual,
//                                             int buckets, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new RotNKDTDataSetIndexation<SSR>(M, B, th);
//    dsi->sweep();

//    const int qual = bnb_search_ml_table<SSR, 8> (*dsi, known_sol_qual, gap,
//                                                  buckets, guessAndResult);
//    delete dsi;

//    //result = Eigen::AngleAxisd(ssrAngle(guessAndResult), ssrAxis(guessAndResult));
//    result = centre(guessAndResult);
//    return qual;
//}





