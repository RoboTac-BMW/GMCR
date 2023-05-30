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

#include "rot3_matches_evaluator.h"
#include "util_sort.h"
#include "reg_binary_tree.h"
#include "reg_common.h"
#include "geometry.h"


inline
void multiply_aux(reg::Matrix3X &Y, reg::Matrix3 &R,  const reg::Matrix3X &X )
{
    char *chn = (char *)"N";
    double alpha = 1.0, beta = 0.0;
    ptrdiff_t m = 3;
    ptrdiff_t n = X.cols();
    
    dgemm(chn, chn, &m, &n, &m, &alpha, R.getPr(), &m, X.x, &m, &beta, Y.x, &m);
}

using namespace reg::search;
using namespace reg::geometry;

template <class SSR>
Rot3MatchesEvaluator<SSR>::Rot3MatchesEvaluator(const reg::Matrix3X &X,
                                                const reg::Matrix3X &Y, double th ):
    M_in(X), B_in(Y), th(th), _size(0)
{
    mxAssert(th>0 && X.cols()==Y.cols(), "invalid input");
}

template <class SSR>
Rot3MatchesEvaluator<SSR>::Rot3MatchesEvaluator()
{}

template <class SSR>
int Rot3MatchesEvaluator<SSR>::size() const
{
    mxAssert(_size>=0, "Wring size");
    return _size;
}

template <class SSR>
int Rot3MatchesEvaluator<SSR>::sweep()
{
    M = M_in;//.topLeftCorner(3, M_in.cols());
    B = B_in;//.topLeftCorner(3, B_in.cols());
    _size = M.cols();
    return _size;
}


template <class SSR>
int Rot3MatchesEvaluator<SSR>::sweep(
        int *matchList, int matchListSize, std::vector<bool> &matches)
{
    mxAssert(matches.size()>=matchListSize,"");
    mxAssert(matchListSize>0 && matchListSize<=M_in.cols(),"");

    // M according to the match list
    M = Matrix3X(matchListSize); //Eigen::Matrix3Xd::Zero(3,matchListSize);
    int j=0;
    for(int i=0; i<matchListSize; i++)
    {
        M(0,j)=M_in(0,matchList[i]);
        M(1,j)=M_in(1,matchList[i]);
        M(2,j)=M_in(2,matchList[i]);

        B(0,j)=B_in(0,matchList[i]);
        B(1,j)=B_in(1,matchList[i]);
        B(2,j)=B_in(2,matchList[i]);

        matches[i]=1;
        j++;
    }
    _size = matchListSize;
    return matchListSize;
}



template <class SSR>
int Rot3MatchesEvaluator<SSR>::evalUpperBound(
        SSR ssr, int lwbnd) const
{
    mxAssert(lwbnd>=0,"");
    //Rotate M and count matches for th+uncertainty

    int bnd=0;
    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    const double delta = ssrMaxAngle(ssr);
    double ang;

    double *y;
    for (int i=0; i<_size && (bnd+_size-i > lwbnd); i++)
    {
        Vector3 x = multiply(R, M.x+3*i);
        y = B.x+3*i;
        
        //obntain angle(m,b)
//        AxisAngle ax;
//        ax = Eigen::Quaterniond().setFromTwoVectors(m, b);
//        ang=ax.angle();

        ang = acos( x.x*y[0] + x.y*y[1]  + x.z*y[2] );
        if (ang<th+delta)
        {
            bnd++;
        }
    }
    return bnd;
}

template <class SSR>
int Rot3MatchesEvaluator<SSR>::evalUpperBound(
        SSR ssr, int lwbnd,
        int *matchList, int matchListSize,
        std::vector<bool> &matches) const
{
    mxAssert(lwbnd>=0 && matchListSize>=0 && matchListSize<=M.cols(),"wrong input");

    int bnd=0;

    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    const double delta = ssrMaxAngle(ssr);
    double ang;
    
    double *y;
    
    for (int i=0; i<matchListSize && (bnd+matchListSize-i > lwbnd); i++)
    {
//        Vector3 m = R*M.col(matchList[i]);
        Vector3 x = multiply(R, M.x + 3*matchList[i]);

        //Vector3 b = B.col(matchList[i]);
        y = B.x+3*matchList[i];
        
//        AxisAngle ax;
//        ax = Eigen::Quaterniond().setFromTwoVectors(m, b);
//        ang=ax.angle();
        ang = acos( x.x*y[0] + x.y*y[1]  + x.z*y[2] );

        if (ang<th+delta)
        {
            bnd++;
            matches[i]=1;
        }
        else
        {
            matches[i]=0;
        }
    }
    return bnd;
}

// Y = R*X;




template <class SSR>
int Rot3MatchesEvaluator<SSR>::evalLowerBound(SSR ssr) const
{
    int qual=0;
    
    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    double ang;

    Matrix3X RM(_size);
    
    multiply_aux(RM, R, M); //RM = R*M;

    double *x, *y;
    for (int i=0; i<_size ; i++)
    {
        x = RM.x + 3*i;
        y = B.x + 3*i;

        ang = acos( x[0]*y[0] + x[1]*y[1]  + x[2]*y[2] );

        if (ang<th)
        {
            qual++;
        }
    }

    return qual;
}

template <class SSR>
int Rot3MatchesEvaluator<SSR>::evalLowerBound(
        SSR ssr, int *matchList, int matchListSize) const
{
    mxAssert(matchListSize>=0 && matchListSize<=M.cols(),"");

    int qual=0;

    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    double ang;
    
    double *y;

    for (int i=0; i<matchListSize; i++)
    {
        Vector3 x = multiply(R, M.x + 3*matchList[i]);
        y = B.x + 3*matchList[i];

        ang = acos( x.x*y[0] + x.y*y[1]  + x.z*y[2] );
        
        if (ang<th)
        {
            qual++;
        }
    }

    return qual;
}


template <class SSR>
int Rot3MatchesEvaluator<SSR>::evalLowerBound(
        SSR ssr, int *matchList, int matchListSize,
        std::vector<bool> &matches) const
{
    mxAssert(matchListSize>=0 && matchListSize<=M.cols(),"wrong input");

    int qual=0;

    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    double ang;

    double *y;
    
    for (int i=0; i<matchListSize; i++)
    {
        Vector3 x = multiply(R, M.x + 3*matchList[i]);
        y = B.x + 3*matchList[i];

        ang = acos( x.x*y[0] + x.y*y[1]  + x.z*y[2] );
        
        if (ang<th)
        {
            qual++;
            matches[i]=1;
        }
        else
        {
            matches[i]=0;
        }
    }
    return qual;
}
