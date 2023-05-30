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

#include "rot3_matches_evaluator_l2.h"
#include "util_sort.h"
#include "reg_binary_tree.h"
#include "reg_common.h"
#include "geometry.h"


inline
void _multiply_aux(reg::Matrix3X &Y, reg::Matrix3 &R, const reg::Matrix3X &X )
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
Rot3MatchesEvaluatorL2<SSR>::Rot3MatchesEvaluatorL2(const reg::Matrix3X &X,
                                                    const reg::Matrix3X &Y, double th ):
    X_in(X), Y_in(Y), th(th), _size(0), offset(0)
{
    mxAssert(th>0 && X.cols()==Y.cols(), "invalid input");
}

template <class SSR>
Rot3MatchesEvaluatorL2<SSR>::~Rot3MatchesEvaluatorL2()
{}

template <class SSR>
int Rot3MatchesEvaluatorL2<SSR>::size() const
{
    mxAssert(_size>=0, "Wrong size");
    return _size;
}

template <class SSR>
int Rot3MatchesEvaluatorL2<SSR>::sweep()
{
    _size = Y_in.cols();
    mxAssert(_size>0, "need non zero size");
    
    // store norm vectors
    X = Matrix3X(_size);
    Y = Matrix3X(_size);
    
    // corresponding ang. errors
    TH = Vector(_size);
    
    int k=0;
    double lenx, leny;
    double d;
    double dev;
    double *inx, *iny, *x, *y;
    for (int i = 0; i<_size; i++)
    {
        inx = X_in.getPr() + 3*i;
        iny = Y_in.getPr() + 3*i;
        
        lenx = sqrt( inx[0]*inx[0] + inx[1]*inx[1] + inx[2]*inx[2] );
        leny = sqrt( iny[0]*iny[0] + iny[1]*iny[1] + iny[2]*iny[2] );
        
        d = fabs(leny - lenx);
        if ( d>th )
        {
            continue;
        }
        
        if (lenx<=DUMMY_PRECISION || leny<=DUMMY_PRECISION)
        {
            offset++;
            continue;
        }
        
        dev = circleintersection(lenx, leny, th);
        
        mxAssert(dev>0,"invalid dev");
        
        if (fabs(dev-PI) <= DUMMY_PRECISION )
        {
            offset++;
            continue;
        }
        
        x = X.getPr() + 3*k;
        y = Y.getPr() + 3*k;
        
        for (int j=0; j<3; j++)
        {
            x[j] = inx[j]/lenx;
            y[j] = iny[j]/leny;
        }
        
        mxAssert(dev<PI, "invalid error");
        TH(k) = dev;
        
        k++;
    }
    _size = k;
    
    mxAssert(_size>=0,"size>=0");
    
    X.setSize(_size);
    Y.setSize(_size);
    TH.setSize(_size);
    
    return _size;
}


template <class SSR>
int Rot3MatchesEvaluatorL2<SSR>::sweep(
        int *matchList, int matchListSize, std::vector<bool> &matches)
{
    mxAssert(matches.size()>=matchListSize,"");
    mxAssert(matchListSize>0 && matchListSize<=X_in.cols(),"");
    
    X = Matrix3X(matchListSize);
    Y = Matrix3X(matchListSize);

    // corresponding ang. errors
    TH = Vector(matchListSize);

    double lenx, leny;
    double d;
    double dev;
    double *inx, *iny, *x, *y;
    
    int k=0;
    for(int i=0; i<matchListSize; i++)
    {
        inx = X_in.getPr() + 3*matchList[i];
        iny = Y_in.getPr() + 3*matchList[i];
        
        lenx = sqrt(inx[0]*inx[0] + inx[1]*inx[1] + inx[2]*inx[2]);
        leny = sqrt(iny[0]*iny[0] + iny[1]*iny[1] + iny[2]*iny[2]);
        
        d = fabs(leny - lenx);
        if ( d>th )
        {
            matches[i]=0;
            continue;
        }
        
        if (lenx<=DUMMY_PRECISION || leny<=DUMMY_PRECISION)
        {
            matches[i]=0;
            offset++;
            continue;
        }
        
        dev = circleintersection(lenx, leny, th);
        
        mxAssert(dev>0,"invalid dev");
        
        if (fabs(dev-PI) <= DUMMY_PRECISION )
        {
            matches[i]=0;
            offset++;
            continue;
        }
        
        x = X.getPr() + 3*k;
        y = Y.getPr() + 3*k;
        
        for (int j=0; j<3; j++)
        {
            x[j] = inx[j]/lenx;
            y[j] = iny[j]/leny;
        }
        
        TH(k) = dev;
       
        matches[i]=1;
        k++;
    }
    
    _size = k;
    
    mxAssert(_size>0,"size>0");
    
    X.setSize(_size);
    Y.setSize(_size);
    TH.setSize(_size);
    
    return _size;
}



template <class SSR>
int Rot3MatchesEvaluatorL2<SSR>::evalUpperBound(
        SSR ssr, int lwbnd) const
{
    mxAssert(lwbnd>=0,"lower bound must be >=0");

    int bnd = offset;
    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    const double delta = ssrMaxAngle(ssr);
    double ang;

    double *y;
    
    for (int i=0; i<_size && (bnd+_size-i > lwbnd); i++)
    {
        Vector3 x = multiply(R, X.x+3*i);
        y = Y.x+3*i;
        
        ang = acos( x.x*y[0] + x.y*y[1] + x.z*y[2] );
        if (ang<=TH(i)+delta)
        {
            bnd++;
        }
    }
    
    mxAssert(bnd<=X_in.n, "wrong upbnd");
    return bnd;
}

template <class SSR>
int Rot3MatchesEvaluatorL2<SSR>::evalUpperBound(
        SSR ssr, int lwbnd,
        int *matchList, int matchListSize,
        std::vector<bool> &matches) const
{
    mxAssert(lwbnd>=0 && matchListSize>=0 && matchListSize<=X.cols(),"wrong input");

    int bnd = offset;

    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    const double delta = ssrMaxAngle(ssr);
    double ang;
    double *y;
    
    for (int i=0; i<matchListSize && (bnd+matchListSize-i > lwbnd); i++)
    {
        Vector3 x = multiply(R, X.x + 3*matchList[i]);

        y = Y.x+3*matchList[i];
        
        ang = acos( x.x*y[0] + x.y*y[1]  + x.z*y[2] );

        if (ang<=TH(matchList[i])+delta)
        {
            bnd++;
            matches[i]=1;
        }
        else
        {
            matches[i]=0;
        }
    }
    
    mxAssert(bnd<=X_in.n, "wrong upbnd");
    return bnd;
}


template <class SSR>
int Rot3MatchesEvaluatorL2<SSR>::evalLowerBound(SSR ssr) const
{
    int qual=offset;
    
    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    double ang;

    Matrix3X RX(_size);
    
    _multiply_aux(RX, R, X); //RX = R*X;

    double *x, *y;
    for (int i=0; i<_size ; i++)
    {
        x = RX.x + 3*i;
        y = Y.x + 3*i;

        ang = acos( x[0]*y[0] + x[1]*y[1] + x[2]*y[2] );
        if (ang<=TH(i))
        {
            qual++;
        }
    }

    mxAssert(qual<=X_in.n, "wrong lwbnd");

    return qual;
}

template <class SSR>
int Rot3MatchesEvaluatorL2<SSR>::evalLowerBound(SSR ssr, int *matchList, int matchListSize) const
{
    mxAssert(matchListSize>=0 ,"");
    mxAssert(matchListSize<=X.cols(),"");

    
    int qual=offset;

    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    double ang;
    double *y;

    for (int i=0; i<matchListSize; i++)
    {
        Vector3 x = multiply(R, X.x + 3*matchList[i]);
        y = Y.x + 3*matchList[i];

        ang = acos( x.x*y[0] + x.y*y[1]  + x.z*y[2] );
        
        if (ang<=TH(matchList[i]))
        {
            qual++;
        }
    }
    
    mxAssert(qual<=X_in.n, "wrong lwbnd");


    return qual;
}


template <class SSR>
int Rot3MatchesEvaluatorL2<SSR>::evalLowerBound(
        SSR ssr, int *matchList, int matchListSize,
        std::vector<bool> &matches) const
{
    mxAssert(matchListSize>=0 && matchListSize<=X.cols(),"wrong input");

    int qual=offset;

    Matrix3 R;
    fromAxisAngle(R, centre(ssr));
    
    double ang;
    double *y;
    
    for (int i=0; i<matchListSize; i++)
    {
        Vector3 x = multiply(R, X.x + 3*matchList[i]);
        y = Y.x + 3*matchList[i];

        ang = acos( x.x*y[0] + x.y*y[1]  + x.z*y[2] );
        
        if (ang<=TH(matchList[i]))
        {
            qual++;
            matches[i]=1;
        }
        else
        {
            matches[i]=0;
        }
    }
    
    mxAssert(qual<=X_in.n, "wrong lwbnd");

    return qual;
}
