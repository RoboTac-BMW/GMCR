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
#include "matrix.h"
#include "mex.h"
#include "blas.h"
#include "math.h"
#include "gore.h"
#include "registration.h"

using namespace reg;
using namespace reg::gore;


// Y = X - t
inline
void translate(double *Y, const double *X, const double *t, int numOfPts)
{
    mxAssert(numOfPts>=0, "invalid number of points");
    for(int i=0; i<numOfPts; i++)
    {
        Y[3*i  ] = X[3*i  ]-t[0];
        Y[3*i+1] = X[3*i+1]-t[1];
        Y[3*i+2] = X[3*i+2]-t[2];
    }
}


// D(i) = || R*X(i)  -  Y(i)  ||^2
inline
int eval(const Matrix3 &R, double *X, double *Y,
         double *D, const int n, const double th, int lwbnd)
{
    mxAssert(n>=0, "invalid number of points");
    mxAssert(th>0, "invalid threshold");
    mxAssert(lwbnd>=0 && lwbnd<=n, "invalid lower bound");
    
    //          [ x1 ]
    //          [ x2 ]
    //          [ x3 ]
    //   [a d g]
    //   [b e h]
    //   [c f i]
    
    
    const double &ra = R.m[0];
    const double &rb = R.m[1];
    const double &rc = R.m[2];
    const double &rd = R.m[3];
    const double &re = R.m[4];
    const double &rf = R.m[5];
    const double &rg = R.m[6];
    const double &rh = R.m[7];
    const double &ri = R.m[8];
    
    int q=0; //number of matches
    double aux;
    const double sqrdth=th*th;
    int k;
    
    for(int i=0; i<n && q+(n-i)>lwbnd; i++)
    {
        k=3*i;
        
        // translate x
        const double &x1 = X[k];
        const double &x2 = X[k+1];
        const double &x3 = X[k+2];
        
        // x-axis
        aux = ra*x1 + rd*x2 + rg*x3 - Y[k];
        D[i] = aux*aux;
        
        if (D[i]>sqrdth)
        {
            continue;
        }
        
        // y-axis
        aux = rb*x1 + re*x2 + rh*x3 - Y[k+1];
        D[i] += aux*aux;
        
        if (D[i]>sqrdth)
        {
            continue;
        }
        
        // z-axis
        aux = rc*x1 + rf*x2 + ri*x3 - Y[k+2] ;
        D[i] += aux*aux;
        
        if (D[i]<=sqrdth)
        {
            q++;
        }
    }

    return q;
}


int _gore6(size_t *H, double *X, double *Y, int numOfPts,
           double th, int lwbnd, Transform3 &Tout, int &outLwbnd)
{
    mxAssert(lwbnd>=0, "invalid lower bound");
    mxAssert(numOfPts>=lwbnd, "invalid number of points");

    // Density in stabbing
    int upBnd;

    // Rotation matrix
    Matrix3 Ropt;
    //Set identity
    double *RoptPr = Ropt.getPr();
    RoptPr[0] = 1; RoptPr[3] = 0; RoptPr[6] = 0;
    RoptPr[1] = 0; RoptPr[4] = 1; RoptPr[7] = 0;
    RoptPr[2] = 0; RoptPr[5] = 0; RoptPr[8] = 1;
    

    // Lower bound for subproblem k.
    int lwbndK;

    // Vector indicating point matches to be tested.
    std::vector<bool> potentialOutlrs(numOfPts, true);
    
    //Euclidean distances
    double *Distance = (double *)mxMalloc(numOfPts*sizeof(double));

    double *Xk = (double *)mxMalloc(3*numOfPts*sizeof(double));
    double *Yk = (double *)mxMalloc(3*numOfPts*sizeof(double));
    size_t *Hrot = (size_t *)mxMalloc(numOfPts*sizeof(size_t));
    
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
        translate(Xk, X, X+k*3, numOfPts);
        translate(Yk, Y, Y+k*3, numOfPts);
        
        //-----------------------------------------------------------------
        //  Obtain upper bound
        //-----------------------------------------------------------------
        
        // RUN GORE
        for(int i=0; i<numOfPts; i++)
        {
            Hrot[i]=i;
        }

        int n2 = gore3_l2(Hrot, Xk, Yk, numOfPts, 2*th, lwbnd, true, Ropt.m, outLwbnd);
        upBnd = n2;
        
        //----------------------------------------
        // Estimate and update lower bound
        //----------------------------------------

        lwbndK = eval(Ropt, Xk, Yk, Distance, numOfPts, th, lwbnd);

        // Update lower bound
        if(lwbndK > lwbnd)
        {
            lwbnd = lwbndK;
            
            // Update array of potential outlrs from position k+1
            for (int i=k+1; i<numOfPts; i++)
            {
                potentialOutlrs[i] = Distance[i]>th*th;
            }
            
            //Update output transformation
            
            Vector3 tyinv(Y[k*3], Y[k*3+1], Y[k*3+2]);
            Vector3 tx(-X[k*3], -X[k*3+1], -X[k*3+2]);
            Transform3 Rtform(Ropt);
            Transform3 tftyinv(tyinv);
            Transform3 tftx(tx);
            
            Tout = tftyinv*Rtform*tftx;
        }

        
        // improve using BnB
        if (upBnd > lwbnd)
        {
            //RUN BnB
            AxisAngle bnbresp;
            Matrix3X Xkaux(n2);
            Matrix3X Ykaux(n2);
            
            for (int i=0;i<n2;i++)
            {
                double *xk=Xk+3*Hrot[i];
                double *yk=Yk+3*Hrot[i];
                
                Xkaux(0,i)=xk[0];
                Xkaux(1,i)=xk[1];
                Xkaux(2,i)=xk[2];
                
                Ykaux(0,i)=yk[0];
                Ykaux(1,i)=yk[1];
                Ykaux(2,i)=yk[2];
            }


            upBnd = reg::search::rot3_matches_l2(Xkaux, Ykaux, 2*th, lwbndK, 0, bnbresp);
            
            //use obtained rotation to update lower bound
            fromAxisAngle(Ropt, bnbresp);
            
            lwbndK = eval(Ropt, Xk, Yk, Distance, numOfPts, th, lwbnd);
            
            // Update lower bound
            if(lwbndK > lwbnd)
            {
                lwbnd = lwbndK;
                
                // Update array of potential outlrs from position k+1
                for (int i=k+1; i<numOfPts; i++)
                {
                    potentialOutlrs[i] = Distance[i]>th*th;
                }
                
                //Update output transformation
                
                Vector3 tyinv(Y[k*3], Y[k*3+1], Y[k*3+2]);
                Vector3 tx(-X[k*3], -X[k*3+1], -X[k*3+2]);
                Transform3 Rtform(Ropt);
                
                Transform3 tftyinv(tyinv);
                Transform3 tftx(tx);
                
                Tout = tftyinv*Rtform*tftx;
            }

        } //improve using bnb
      
        
        //------------------------------------------------
        // Check upper bound estimation
        //------------------------------------------------
        if (upBnd < lwbnd)
        {
            mxAssert(numOfPts-1>=0,"wrong index");
           
            numOfPts--;
            
            if(k<numOfPts)
            {
                const int endIdx = 3*numOfPts;
                std::copy(X+endIdx, X+endIdx+3, X+3*k);
                std::copy(Y+endIdx, Y+endIdx+3, Y+3*k);
                H[k] = H[numOfPts];
                potentialOutlrs[k] = potentialOutlrs[numOfPts];
            }
            
            mxAssert(numOfPts>=lwbnd, "");
            continue;
        }

        // Move to the next correspondence to be tested
        k++;
    } // End iteration over correspondences

    mxFree(Xk);
    mxFree(Yk);
    mxFree(Distance);
  //  mxDestroyArray(idx_gorel2);
    mxFree(Hrot);

    outLwbnd = lwbnd;

    // number of removed points
    return numOfPts;
}


int _gore6_rep(size_t *H, double *X, double *Y, int numOfPts,
               double th, int lwbnd, Transform3 &Tout, int &outLwbnd)
{
    mxAssert(numOfPts>0,"invalid number of points");
    mxAssert(lwbnd>=0,"invalid lower bound");
    mxAssert (th>0,"invalid threshold");


    int n = _gore6(H,X,Y,numOfPts,th,lwbnd,Tout,outLwbnd);
    mxAssert(outLwbnd>=lwbnd, "");

    int bestLwbnd = outLwbnd;

    Transform3 T;

    while (n < numOfPts)
    {
        std::cout<<"N' "<<n<<std::endl;
        numOfPts = n;
        
        n = _gore6(H,X,Y,numOfPts,th,bestLwbnd,T,outLwbnd);
        
        //Update the best solution
        if (outLwbnd > bestLwbnd)
        {
            bestLwbnd = outLwbnd;
            Tout = T;
        }
    }
    mxAssert(n==numOfPts, "");

    outLwbnd = bestLwbnd;
    return numOfPts;
}


int reg::gore::gore6(size_t *H, const double *Xin, const double *Yin,
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
    
    int n;
    if (!repFlag)
    {
        n = _gore6(H,X,Y,numOfPts,th,lwbnd,T,outLwbnd);
    }
    else
    {
        n = _gore6_rep(H,X,Y,numOfPts,th,lwbnd,T,outLwbnd);
    }
    
    mxFree(X);
    mxFree(Y);
    
    mxAssert(numOfPts-n>=0, "");
    mxAssert(outLwbnd>=lwbnd, "");
    
    return n;
}
