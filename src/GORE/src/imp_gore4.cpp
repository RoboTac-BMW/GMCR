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

#include <string.h>
#include <vector>
#include <algorithm>  //sort
#include "matrix.h"
#include "mex.h"
#include "blas.h"
#include "math.h"
#include "gore.h"



using namespace reg;

inline
double circleintersection(double R, double d, double r)
{
    mxAssert(R>=0 && d>=0 && r>=0, "parametres must be positive");
    mxAssert(d<(R+r),"no intersecting points");
    // Return value is between 0 and pi.
    
    double rat, x, angle;
    
    if (d<=DUMMY_PRECISION)
    {
        return PI;
    }
    
    
    x = (d*d-r*r+R*R)/(2.0*d);
    
    rat = x/R;
    if (rat<=-1.0)
    {
        return PI;
    }
    
    angle= acos(rat);
    mxAssert(angle<=PI, "angle must be < PI");
    return angle;
}



inline
void rotationFromAngle(const double theta, Matrix2 &R)
{
    double *r = R.getPr();
    r[0] = r[3] = cos(theta);
    r[1] = sin(theta);
    r[2] = -r[1];
}


int eval(const Matrix2 &R, const Vector3 &tx, const Vector3 &ty, double*X, double *Y,
            double *D, int n, const double th, const int lwbnd)
{
    //          [ x1 ]
    //          [ x2 ]
    //   [a c]
    //   [b d]
    
    mxAssert(n>0,"");
    mxAssert(th>0,"");

    
    const double &ra = R.m[0];
    const double &rb = R.m[1];
    const double &rc = R.m[2];
    const double &rd = R.m[3];
    
    
    int q=0; //number of matches

    double x1, x2;
    double aux;
    
    const double sqrdth=th*th;
    int k;
    
    const double dz = tx.z-ty.z;
    for(int i=0; i<n && q+(n-i)>lwbnd; i++)
    {
        k=3*i;
        
        //z-axis
        D[i] = X[k+2] - Y[k+2] + dz;
        D[i] *= D[i];
        
        if (D[i]>sqrdth)
        {
            continue;
        }
        
        //translate x
        x1 = X[k] + tx.x;
        x2 = X[k+1] + tx.y;
        
        //x-axis
        aux = ra*x1 + rc*x2 - Y[k] - ty.x;
        D[i]+=aux*aux;
        
        if (D[i]>sqrdth)
        {
            continue;
        }
        
        //y-axis
        aux = rb*x1 + rd*x2 - Y[k+1] - ty.y;
        D[i]+=aux*aux;
        
        if (D[i]<=sqrdth)
        {
            q++;
        }
    }

    return q;
}


/**
 * @brief Structure to be sorted by the stabbing method.
 */
struct LimitLabel
{
    double value;
    short label;  //label in {1,-1}
};


/**
 * @brief Operator to sort LimitLabel instances.
 */
struct limitLabelComparator
{
    bool operator()(const LimitLabel &v1, const LimitLabel &v2) const
    {
        return  v1.value < v2.value;
    }
};


int _gore4(size_t *H, double *X, double *Y, int numOfPts,
           double th, int lwbnd, Transform3 &Tout, int &outLwbnd)
{
    mxAssert(lwbnd>=0, "invalid lower bound");
    mxAssert(numOfPts>=lwbnd, "invalid number of points");

    // Density in stabbing
    int dsty, maxDsty;

    // Counters
    int numberOfLimits, numOfInsertions;

    // Rotation matrix
    Matrix2 Ropt;

    // Stabbing vars. to control iterations.
    int i, offset;

    // Stabbing opt. val.
    double opt;

    int optIdx;

    // Lower bound for subproblem k.
    int lwbndK;

      // Array to be sorted for the stabbing algorithm
    struct LimitLabel *ll = new struct LimitLabel[3*numOfPts];

    // Vector indicating point matches to be tested.
    std::vector<bool> potentialOutlrs(numOfPts, true);
    
    //Euclidean distances
    double *Distance = (double *)mxMalloc(numOfPts*sizeof(double));

    
   // double *x, *y;
    double xlen, ylen, dev, dz, d, rth, xazi, yazi, beg, end;

    double x[3];
    double y[3];
    double *xpr, *ypr;
    
    // Iterate to try to reject matches.
    for (int k=0; k<numOfPts; )
    {
        // Find a potential outlier to reject
        while (k<numOfPts && !potentialOutlrs[k])
        {
            k++;
        }
        
        if (k==numOfPts)
        {
            break;
        }


        // Translate visited correspondence to the origin
        const double *txPr = X+k*3;
        const double *tyPr = Y+k*3;
        
        Vector3 tx(-txPr[0], -txPr[1], -txPr[2]);
        Vector3 ty(-tyPr[0], -tyPr[1], -tyPr[2]);
        
        //-----------------------------------------------------------------
        //  Stabbing on rotation angle about the Z axis
        //-----------------------------------------------------------------
        offset = 1; // Initialise for i==k
        numberOfLimits = numOfInsertions = 0;

        for (i=0; i<numOfPts && 
                  (offset+numOfInsertions+(numOfPts-i))>=lwbnd; i++)
        {
            // skip point match k
            if (i==k)
            {
                continue;
            }

            xpr = X+3*i;
            ypr = Y+3*i;
            
            // Translate to origin
            x[0]=xpr[0] + tx.x;
            x[1]=xpr[1] + tx.y;
            x[2]=xpr[2] + tx.z;
            
            y[0]=ypr[0] + ty.x;
            y[1]=ypr[1] + ty.y;
            y[2]=ypr[2] + ty.z;
            
            
            dz = fabs(x[2]-y[2]);
            if ( dz > 2.0*th )
            {
                continue;
            }
            
            xlen = sqrt( x[0]*x[0] + x[1]*x[1] );
            ylen = sqrt( y[0]*y[0] + y[1]*y[1] );
            
            
            d = fabs(ylen - xlen); /*d = fabs(blen-MLEN[i]);*/
            rth = sqrt(4*th*th - dz*dz);  //(2*th2)^2
            if ( d>rth )  /* |d|>rth  */
            {
                continue;
            }
            
            if (xlen<=DUMMY_PRECISION)
            {
                offset++;
                continue;
            }
            
            dev = circleintersection(xlen, ylen, rth);
            
            mxAssert(dev>0,"invalid dev");
            
            if (fabs(dev-PI) <= DUMMY_PRECISION )
            {
                offset++;
                continue;
            }
            
            xazi = atan2(x[1], x[0]);
            yazi = atan2(y[1], y[0]);
            
//            double dot = x[0]*y[0] + x[1]*y[1];
//            double theta = acos(dot);
//            if (theta<0)
//            {
//                std::cout<<"theta = "<<180*theta/PI<<std::endl;
//            }
            
            
            beg = fmod(yazi-dev-xazi, TWOPI);
            if (beg<0)
            {
                beg += TWOPI;
            }
            
            end = fmod(yazi+dev-xazi, TWOPI);
            if (end<0)
            {
                end += TWOPI;
            }
            
            mxAssert(beg>=0 && beg<=TWOPI , "wrong angle");
            mxAssert(end>=0 && end<=TWOPI , "wrong angle");
            
            if (end>=beg)
            {
                ll[numberOfLimits].value = beg;  ll[numberOfLimits++].label = 1;
                ll[numberOfLimits].value = end;  ll[numberOfLimits++].label =-1;
                numOfInsertions++;
            }
            else
            {
                ll[numberOfLimits].value = 0;      ll[numberOfLimits++].label = 1;
                ll[numberOfLimits].value = end;    ll[numberOfLimits++].label =-1;
                ll[numberOfLimits].value = beg;    ll[numberOfLimits++].label = 1;
                ll[numberOfLimits].value = TWOPI;  ll[numberOfLimits++].label =-1;
                numOfInsertions++;
            }

        } // End iterate over intervals

        if ( numOfInsertions+offset < lwbnd )
        {
            numOfPts--;
            if(k<numOfPts)
            {
                const int endIdx = 3*numOfPts;
                std::copy(X+endIdx, X+endIdx + 3, X+3*k);
                std::copy(Y+endIdx, Y+endIdx + 3, Y+3*k);
                H[k] = H[numOfPts];
                potentialOutlrs[k]=potentialOutlrs[numOfPts];
            }
            
            mxAssert(numOfPts>=lwbnd, "");
            continue;
        }

        //-----------------------------------
        // Obtain max. intersection density
        //-----------------------------------
        std::sort(ll, ll+numberOfLimits, limitLabelComparator());

        dsty = maxDsty = offset;
        optIdx=0;
        for(i=0; i<numberOfLimits; i++)
        {
            dsty += ll[i].label;
            if(dsty > maxDsty)
            {
                maxDsty = dsty;
                optIdx = i;
            }
        }

        //----------------------------------------
        // Estimate and update lower bound
        //----------------------------------------
        opt = .5*(ll[optIdx].value + ll[optIdx+1].value);
        
        rotationFromAngle(opt, Ropt);


        lwbndK = eval(Ropt, tx, ty, X, Y, Distance, numOfPts, th, lwbnd);

        // Update lower bound
        if(lwbndK > lwbnd)
        {
            lwbnd = lwbndK;

            // Update array of potential outlrs from position k+1
            for (i=k+1; i<numOfPts; i++)
            {
                potentialOutlrs[i] = Distance[i]>th*th;
            }

            //Update output transformation
            Vector3 tyinv(-ty.x, -ty.y, -ty.z);
            Transform3 Rtform(Ropt);
            Transform3 tftyinv(tyinv);
            Transform3 tftx(tx);
            
            Tout = tftyinv*Rtform*tftx;
            
        }

        //------------------------------------------------
        // Check upper bound estimation
        //------------------------------------------------
        if (maxDsty < lwbnd)
        {
            numOfPts--;
            if(k<numOfPts)
            {
                const int endIdx = 3*numOfPts;
                std::copy(X+endIdx, X+endIdx + 3, X+3*k);
                std::copy(Y+endIdx, Y+endIdx + 3, Y+3*k);
                H[k] = H[numOfPts];
                potentialOutlrs[k]=potentialOutlrs[numOfPts];
            }
            
            mxAssert(numOfPts>=lwbnd, "");
            continue;
        }

        // Move to the next correspondence to be tested
        k++;
    } // End iteration over correspondences

    delete []ll;
    
    mxFree(Distance);

    outLwbnd = lwbnd;

    // number of removed points
    return numOfPts;
}


int _gore4_rep(size_t *H, double *X, double *Y, int numOfPts,
               double th, int lwbnd, Transform3 &Tout, int &outLwbnd)
{
    mxAssert(numOfPts>0,"");
    mxAssert(lwbnd>=0,"");
    mxAssert (th>0,"invalid threshold");

    int n = _gore4(H,X,Y,numOfPts,th,lwbnd,Tout,outLwbnd);
    mxAssert(outLwbnd>=lwbnd, "");
    
    int bestLwbnd = outLwbnd;

    Transform3 T;

    while (n < numOfPts)
    {
        std::cout<<"N' "<<n<<std::endl;
        numOfPts = n;

        n = _gore4(H,X,Y,numOfPts,th,bestLwbnd,T,outLwbnd);
        
        //Update the best solution
        if (outLwbnd > bestLwbnd)
        {
            bestLwbnd = outLwbnd;
            Tout = T;
        }

    } //while

    outLwbnd = bestLwbnd;
    return numOfPts;
}


int reg::gore::gore4(size_t *H, const double *Xin, const double *Yin,
                     int numOfPts, double th, int lwbnd,
                     bool repFlag, Transform3 &T, int &outLwbnd)
{
    mxAssert(numOfPts>0, "invalid number of points");
    mxAssert(lwbnd>=0, "invalid lower bound");
    mxAssert(th>0,"invalid threshold");
    
    double *X = (double *)mxMalloc(3*numOfPts*sizeof(double));
    double *Y = (double *)mxMalloc(3*numOfPts*sizeof(double));
    
    std::copy(Xin, Xin+3*numOfPts, X);
    std::copy(Yin, Yin+3*numOfPts, Y);
    
    int remOut;
    if (!repFlag)
    {
        remOut = _gore4(H,X,Y,numOfPts,th,lwbnd,T,outLwbnd);
    }
    else
    {
        remOut = _gore4_rep(H,X,Y,numOfPts,th,lwbnd,T,outLwbnd);
    }
    
    mxFree(X);
    mxFree(Y);
    
    mxAssert(outLwbnd>=lwbnd, "wrong result");
    mxAssert(remOut>=0, "rem. out must be non negative");
    return remOut;
}
