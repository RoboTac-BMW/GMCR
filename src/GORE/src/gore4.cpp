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

#include "gore.h"

// Input
#define X_IN      prhs[0]
#define Y_IN      prhs[1]
#define TH_IN     prhs[2]
#define LWBND_IN  prhs[3]
#define REPEAT_IN prhs[4] //If 1, repeat until no outliers can be removed.
// Output
#define H_OUT     plhs[0]
#define T_OUT     plhs[1]
#define LWBND_OUT plhs[2]

using namespace reg;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nlhs != 3)
    {
        mexErrMsgTxt("Invalid number of output variables. Expected 3 LHS vars.");
    }

    if (nrhs != 5)
    {
        mexErrMsgTxt("Invalid number of parameters. Expected 5 parameters.");
    }

    const double th = *mxGetPr(TH_IN);
    const int lwbnd = (int)*mxGetPr(LWBND_IN);
    const int numOfPts = (int)mxGetN(X_IN);
    const bool repeat = (bool)*mxGetPr(REPEAT_IN);;

    if (!mxIsDouble(X_IN) || !mxIsDouble(Y_IN))
    {
        mexErrMsgTxt("Input argument must be of type double.");
    }
    
    if (mxGetM(X_IN) != 3)
    {
        mexErrMsgTxt("Invalid input. Expected 3xn matrices");
    }

    if (mxGetM(Y_IN) != 3)
    {
        mexErrMsgTxt("Invalid input. Expected 3xn matrices");
    }


    int outLwbnd;
    Transform3 T;
    
    size_t *H = (size_t *)mxMalloc(numOfPts*sizeof(size_t));
    for(int i=0; i<numOfPts; i++)
    {
        H[i]=i;
    }
    
    const int n = reg::gore::gore4(H,mxGetPr(X_IN),mxGetPr(Y_IN),numOfPts,
                                   th,lwbnd,repeat,T,outLwbnd);
 
    mxAssert(outLwbnd>=lwbnd, "");
    
    H_OUT = mxCreateNumericMatrix(n, 1, mxUINT32_CLASS, mxREAL);
    
    uint32_T *hpr = (uint32_T *)mxGetData(H_OUT);
    
    for(int i=0; i<n; i++)
    {
        hpr[i] = (uint32_T) (1+H[i]); //convert to matlab indices
    }
    
    mxFree(H);


    LWBND_OUT = mxCreateDoubleScalar(outLwbnd);
    
    T_OUT = mxCreateDoubleMatrix(4, 4, mxREAL);
    std::copy(T.getPr(), T.getPr()+16, mxGetPr(T_OUT));
}
