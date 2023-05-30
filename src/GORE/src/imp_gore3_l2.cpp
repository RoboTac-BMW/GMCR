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

// GORE with L2 error

#include "gore.h"
#include <vector>
#include "matrix.h"
#include "mex.h"
#include "math.h"
#include <iostream>

//Windows
#include <algorithm> //for sort
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DUMMY_PRECISION 1e-12
#define PI  M_PI
#define TWOPI  2*M_PI

//20 degrees
#define MINTH 0.3491


#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)


/**
 * @brief transpose Compute the transpose of a rotation matrix.
 * @param R Rotation matrix.
 * @return A pointer to the rotation matrix after it is transposed.
 */
inline
double *transpose(double *R)
{
    double tmp;
    tmp = R[1]; R[1]=R[3]; R[3]=tmp; //swap
    tmp = R[2]; R[2]=R[6]; R[6]=tmp;
    tmp = R[5]; R[5]=R[7]; R[7]=tmp;
    return R;
}


inline
double circleIntersectionAngle(double R, double d, double r)
{
    mxAssert(R>0 && d>0 && r>0, "parametres must >0");
    
    
    //assert(d<(R+r));
    // Return value is between 0 and pi.
    
    double rat, x, angle;
    
    if (d<=DUMMY_PRECISION)
    {
        return PI;
    }
    
    if( fabs(d-(R+r))<DUMMY_PRECISION )
    {
        return 0;
    }
    
    x = (d*d-r*r+R*R)/(2*d);
    
    rat = x/R;
    if (rat<=-1.0)
    {
        return PI;
    }
    
    angle= acos(rat);
    mxAssert(angle>=0 && angle<=PI, "angle must be < PI");
    return angle;
}


/**
 * @brief rotationFromAngle Obtain rotation matrix about the z-axis.
 * @param theta Rotation angle.
 * @param R Rotation matrix about the z-axis with rotation angle theta.
 */
inline
void rotationFromAngle(const double theta, double *R)
{
    const double s = sin(theta);
    R[0]=R[4]=cos(theta);
    R[1]=s;
    R[2]=R[5]=R[6]=R[7]=0.0;
    R[3]=-s;
    R[8]=1.0;
}


/**
 * @brief Obtain a rotation matrix that aligns a with the z-axis.
 * @param a 3D unit-norm vector.
 * @param R Obtained rotation matrix.
 * @return False if c is too close to the [0 0 -1] where the algorithm is
 * undefined.
 */
inline
bool rotationFromVector(const double *a, double *R)
{
    const double &c = a[2]; //dot product with north pole [0 0 1]
    
    if (c < -1.0 + DUMMY_PRECISION)
    {
        return false;
    }
    
    const double s = sqrt( (1.0+c)*2.0 );
    const double invs = 1.0/s;
    const double x = a[1]*invs;
    const double y = -a[0]*invs;
    const double w = s*0.5;
    
    //Convert quaternion to rotation matrix
    const double tx  = 2.0*x;
    const double ty  = 2.0*y;
    const double twx = tx*w;
    const double twy = ty*w;
    const double txx = tx*x;
    const double tyy = ty*y;
    
    R[0] = 1.0-tyy;
    R[1] = R[3] = ty*x;
    R[2] = -twy;
    R[4] = 1.0-txx;
    R[5] = twx;
    R[6] = twy;
    R[7] = -twx;
    R[8] = 1.0-(txx+tyy);
    
    return true;
}


// Compute T = R*S
// R,T,S are 3x3 matrices
inline
void matrix3Multiply(double *T, double *R, double *S)
{
    T[0]=R[0]*S[0]+R[3]*S[1]+R[6]*S[2]; T[3]=R[0]*S[3]+R[3]*S[4]+R[6]*S[5]; T[6]=R[0]*S[6]+R[3]*S[7]+R[6]*S[8];
    T[1]=R[1]*S[0]+R[4]*S[1]+R[7]*S[2]; T[4]=R[1]*S[3]+R[4]*S[4]+R[7]*S[5]; T[7]=R[1]*S[6]+R[4]*S[7]+R[7]*S[8];
    T[2]=R[2]*S[0]+R[5]*S[1]+R[8]*S[2]; T[5]=R[2]*S[3]+R[5]*S[4]+R[8]*S[5]; T[8]=R[2]*S[6]+R[5]*S[7]+R[8]*S[8];
}

inline
void rotatePoint(double *y, double *R, double *x)
{
    y[0] = R[0]*x[0] + R[3]*x[1] + R[6]*x[2];
    y[1] = R[1]*x[0] + R[4]*x[1] + R[7]*x[2];
    y[2] = R[2]*x[0] + R[5]*x[1] + R[8]*x[2];
}


inline
int eval(double *R, double *X, double *Y, double *XLen, double *YLen, double *Rx, double *Ry,
         long int n,
         const double th, int lwbnd)
{
    mxAssert(n>=0,"");
    mxAssert(th>0,"");
    mxAssert(th>0,"");
    
    double d;
    double D;
    double x[3], y;
    
    int q=0; //number of matches
    // Squared Euclidean threshold
    const double eTh = th*th;
    
    for(int i=0; i<n && q+(n-i)>lwbnd; i++)
    {
        //rotate x
        rotatePoint(x, Rx, X+3*i);
        
        const double* ypr = Y+3*i;
        
        y = YLen[i]*(Ry[0]*ypr[0] + Ry[3]*ypr[1] + Ry[6]*ypr[2]);
        d = XLen[i]*(R[0]*x[0] + R[3]*x[1] + R[6]*x[2]) - y;
        D = d*d;
        if (D>eTh)
        {
            continue;
        }
        
        y = YLen[i]*(Ry[1]*ypr[0] + Ry[4]*ypr[1] + Ry[7]*ypr[2]);
        d = XLen[i]*(R[1]*x[0] + R[4]*x[1] + R[7]*x[2]) - y;
        D += d*d;
        if (D>eTh)
        {
            continue;
        }
        
        y = YLen[i]*(Ry[2]*ypr[0] + Ry[5]*ypr[1] + Ry[8]*ypr[2]);
        d = XLen[i]*(R[2]*x[0] + R[5]*x[1] + R[8]*x[2]) - y;
        D += d*d;
        
        if (D<=eTh)
        {
            q++;
        }
    }
    
    return q;
}


//compute squared residuals
//D
inline
void residuals(int offset, double *R, double *X, double *Y, double *XLen, double *YLen, double *Rx, double *Ry,
         double *D, long int n)
{
    mxAssert(n>=0,"");
    
    double d;
    double x[3], y;
    
    // Squared Euclidean threshold
    
    for(int i=offset; i<n;i++)// && q+(n-i)>lwbnd; i++)
    {
        //rotate x
        rotatePoint(x, Rx, X+3*i);
        
        const double* ypr = Y+3*i;
        
        y = YLen[i]*(Ry[0]*ypr[0] + Ry[3]*ypr[1] + Ry[6]*ypr[2]);
        d = XLen[i]*(R[0]*x[0] + R[3]*x[1] + R[6]*x[2]) - y;
        D[i] = d*d;
        
        y = YLen[i]*(Ry[1]*ypr[0] + Ry[4]*ypr[1] + Ry[7]*ypr[2]);
        d = XLen[i]*(R[1]*x[0] + R[4]*x[1] + R[7]*x[2]) - y;
        D[i] += d*d;
        
        y = YLen[i]*(Ry[2]*ypr[0] + Ry[5]*ypr[1] + Ry[8]*ypr[2]);
        d = XLen[i]*(R[2]*x[0] + R[5]*x[1] + R[8]*x[2]) - y;
        D[i] += d*d;
        
    
    }
    
    
}




/**
 * @brief feasibleRegrionAngles return upper and lower angles to bound the
 * uncertainty region.
 * @param z Z coordinate of x.
 * @param xinc Inclination of x.
 * @param cosDazi Cosine of the azimuthal difference between x and y.
 * @param th Angular threshold.
 * @param u Output angle bounding the upper region.
 * @param l Output angle bounding the lower region.
 */
void feasibleRegrionAngles(double z, double xinc, double cosDazi,
                           double epsk, double &u, double &l)
{
    mxAssert(z>=-1 && z<=1,"");     // z coordinate of m
    mxAssert(xinc>=0 && z<=PI,"");  // inclination angle of m
    mxAssert(cosDazi>=-1 && cosDazi<=1,"");//cos of azim diff between m and b
    
    //-----------------------------------
    // Border conditions
    //-----------------------------------

    // m Inclination in the xy-plane: tan(th).
    const double m = tan(epsk);
    mxAssert(m>=-1 && m<=1,"");     // tan(th) --> inclination of the rect

    // Same azimuth
    if (fabs(cosDazi-1.0)<DUMMY_PRECISION)
    {
        l=u=0;
        return;
    }
    
    // Oposite azimuth
    if (fabs(cosDazi+1.0)<DUMMY_PRECISION)
    {
        l=u=2*epsk;
        if(xinc-u<0)
        {
            u=xinc;
        }
        if(xinc+l>PI)
        {
            l=PI-xinc;
        }
        return;
    }
    
    double d = sqrt(1.0-z*z)*cosDazi; // rad. of circ defiend by x
    double md = m*d;
    double c = 1.0 + m*m; //denom quad equation
    double x;
    
    //-----------------------------------
    // Upper region
    //-----------------------------------
    
    // Solve intersection x
    x = ( m*(md-z) + sqrt(c-(z-md)*(z-md)) ) / c;
    
    // Compute theta (polar coordinate) and u = xinc-theta
    if(x>0)
    {
        if (x > d-z/m) // y>0
        {
            // theta = asin(x)
            mxAssert(asin(x)>=0 && asin(x) <= xinc+DUMMY_PRECISION,"");
            u = MAX(xinc-asin(x), 0.0);
        }
        else
        {
            // theta = PI-asin(x)
            mxAssert((PI-asin(x))>=0 && (PI-asin(x)) <= xinc+DUMMY_PRECISION,"");
            u = MAX(xinc-(PI-asin(x)), 0.0);
        }
    }
    else
    {
        // theta=0;
        u=xinc;
    }
    
    //-----------------------------------
    // Lower region
    //-----------------------------------
    
    // Solve intersection for (-m)
    x = ( m*(md+z) + sqrt(c -(z+md)*(z+md)) ) / c;
    
    // Compute theta and l = theta-xinc
    if(x>0)
    {
        if (x<d+z/m) // y>0
        {
            // theta = asin(x)
            mxAssert(asin(x)>=0 && asin(x)>=xinc-DUMMY_PRECISION,"");
            l = MAX(asin(x)-xinc, 0.0);
        }
        else
        {
            // theta = PI-asin(x)
            mxAssert( (PI-asin(x)) >=0 && (PI-asin(x))>=xinc-DUMMY_PRECISION,"");
            l = MAX(PI-asin(x)-xinc, 0.0);
        }
    }
    else
    {
        // theta = PI
        l = PI-xinc;
    }
    
    mxAssert(u>=0 && u<=2*epsk,"");
    mxAssert(l>=0 && l<=2*epsk,"");
}


/**
 * @brief bndAlpha Computes alpha angle for (theta-gamma) in the [0,pi]
 * interval.
 * @param rho theta-gamma
 * @param xinc Inclination of x.
 * @param th Angular threshold.
 * @return angle alpha
 */
double bndAlpha(double rho, double xinc, double th)
{
    mxAssert (rho>=0 && rho<=PI,"");
    mxAssert(xinc>=0 && xinc<=PI,"");
    mxAssert(th>=0 && th<=PI/2.0,"");
    
    const double w = 1.0/sin(xinc);
    double m,b;
    if (rho > asin(w*sin(th)) )
    {
        m = -2.0*sin(0.5*th)*w ;
        b = ( 2.0*rho*sin(0.5*th)+th)*w ;
    }
    else
    {
        m = 2.0*sin(0.5*th)*w ;
        b = ( -2.0*rho*sin(0.5*th)+th)*w ;
    }
    
    double alpha = b/(2.0*sqrt(2.0)/PI - m);
    
    if(alpha>=0 && alpha<=PI/4.0)
    {
        mxAssert(alpha>=0 && alpha<=PI/4.0,"");
        return alpha;
    }
    
    // Intersect second segment
    alpha = (sqrt(2.0)-1.0-b)/(m+2.0*(sqrt(2.0)-2.0)/PI);
    
    
//    double delta = 2*(rho - alpha)*sin(0.5*th) + th;
//    alpha = asin(delta/sin(xinc));
//    
    mxAssert(alpha>=PI/4.0 && alpha<=PI/2.0,"");
    return alpha;
}


/**
 * Bound angle beta for rho in the [0,pi] interval.
 * @brief bndBeta Computes beta angle for (theta+gamma) in the [0,pi]
 * interval.
 * @param rho theta+gamma
 * @param xinc Inclination of x.
 * @param th Angular threshold.
 * @return Beta angle
 */
inline
double bndBeta(double rho, double xinc, double th)
{
    mxAssert(rho>=0 && rho<=PI,"");
    mxAssert(xinc>=0 && xinc<=PI,"");
    mxAssert(th>=0 && th<=PI/2.0,"");
    
    const double w = 1.0/sin(xinc);
    
    const double m = 2.0*sin(0.5*th)*w;
    const double b = (2.0*rho*sin(0.5*th)+th)*w;
    
    double beta = b/(2.0*sqrt(2.0)/PI - m);
    
    mxAssert(beta>=0,"");
    
    if(beta>=0 && beta<=PI/4.0)
    {
        mxAssert(beta>=0 && beta<=PI/4.0,"");
        return beta;
    }
    
    // Intersect second segment
    beta = (sqrt(2.0)-1.0-b)/(m+2.0*(sqrt(2.0)-2.0)/PI);
    
    
//    double delta = 2*(rho + beta)*sin(0.5*th) + th;
//    beta = asin(delta/sin(xinc));
    
    
    mxAssert(beta>=PI/4.0 && beta<=PI/2.0,"");
    return beta;
}


/**
 * @brief uncertaintyAngles Find the parametres alpha and beta for the
 * uncertainty interval [theta-alpha, theta+beta].
 * @param gamma Uncertainty angle containing S_eps(y).
 * @param theta Rotation angle about the z axis to align x to y.
 * @param xinc Inclination of x.
 * @param th Angular threshold.
 * @param alpha Left angle defining the uncertainty
 * interval [theta-alpha, theta+beta].
 * @param beta Right angle defining the uncertainty
 * interval [theta-alpha, theta+beta].
 * @return false if angles alpha and beta could not be obtained.
 */
bool uncertaintyAngles(double gamma, double theta, double xinc, double th,
                       double &alpha, double &beta)
{
    mxAssert(gamma>=0 && gamma<=PI,"");
    mxAssert(theta>=0 && theta<=TWOPI,"");
    mxAssert(xinc>=0 && xinc<=PI,"");
    mxAssert(th>=0 && th<=PI/2.0,"");
    
    //------------------------------------------------------
    //   Solve for beta
    //------------------------------------------------------
    double rho = theta+gamma;
    
    if(rho<=PI)
    {
        mxAssert(rho>0,"");
        
        // Check if there is no solution
        if( sin(xinc) <= (th+(2*rho+PI)*sin(0.5*th)) )
        {
            return false;
        }
        
        beta = bndBeta(rho, xinc, th);
        
        // Fix beta if it is grather than the max. possible solution.
        double aux = sin(TWOPI*sin(0.5*th)+th) / sin(xinc);
        mxAssert(aux>=0,"");
        if(aux <= 1.0)
        {
            beta = MIN(beta, asin(aux));
        }
        
    }
    else // rho>pi
    {
        // Solve as alpha of 2*pi-rho
        rho = TWOPI-rho;
        beta = bndAlpha(rho, xinc, th);
    }
    
    mxAssert(0<=beta && beta <= PI/2.0,"");
    
    //------------------------------------------------------
    //   Solve for alpha
    //------------------------------------------------------
    rho = theta-gamma;
    
    if(rho>0)
    {
        // Check if there is not solution
        if( sin(xinc) <= (th+(2.0*rho+PI)*sin(0.5*th)) )
        {
            return false;
        }
        
        alpha = bndAlpha(rho, xinc, th);
    }
    else
    {
        // Solve as beta for |rho|
        rho = -rho;
        alpha = bndBeta(rho, xinc, th);
    }
    
    mxAssert(0<=alpha && alpha>=asin(sin(th)/sin(xinc)) && alpha <= PI/2.0,"");
    
    return true;
}


/**
 * @brief angularBoundingInterval Find uncertainty interval [l r]
 * for estimate rotation theta.
 * @param theta rotation angle about the z axis to align x to y.
 * @param xinc Inclination of x.
 * @param yinc Inclination of y.
 * @param th Angular threshold.
 * @param l Left value in [l, r].
 * @param r Right value in [l, r].
 * @return false if the interval [l r] could not be obtained.
 */
bool angularBoundingInterval(double theta,double xinc, double yinc,
                             double epsk, double epsi, double &l, double &r)
{
    mxAssert(0<=theta && theta<=TWOPI,"");
    mxAssert(xinc>=0 && xinc<=PI,"");
    mxAssert(yinc>=0 && yinc<=PI,"");
    //mxAssert(epsk>=0 && epsk<=MINTH,"");
    mxAssert(epsi>=0 && epsi<=MINTH,"");

    //TODO: check!!!
    if ( sin(epsi)/sin(yinc) >1 )
    {
        return false;
    }
    
    double alpha, beta;
    const double gamma = asin( sin(epsi)/sin(yinc) );


    mxAssert(gamma >=0 && gamma<=.5*PI, "");
    
    if(theta<PI)
    {
        if(!uncertaintyAngles(gamma, theta, xinc, epsk, alpha, beta))
        {
            return false;
        }
    }
    else
    {
        if (!uncertaintyAngles(gamma, TWOPI-theta, xinc, epsk, beta, alpha))
        {
            return false;
        }
    }
    
    l = theta - gamma - alpha;
    r = theta + gamma + beta;
    
    return true;
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



//Return the new size of H
int _gore_l2(size_t *H, double *X, double *Y, double *XLen, double *YLen,
                int numOfPts, double th, double *TH, int lwbnd, double *Rout, int &outLwbnd)
{
   // mxAssert(numOfPts>=lwbnd, "");
    
    // Euclidean threshold
    const double sqrdTh = th*th;
    
    // Uncertainty angle interval limits
    double thetaL, thetaR;
    
    // Aux. vars. to fix angles to lie in [0,2pi]
    double segAEnd, segBSrt;
    
    // Estimated angle in [0,2pi] about the Z axis.
    double theta;
    
    // Dot product of x and y.
    double dot;
    
    // Angular distance in [0,pi].
    double angdist;
    
    // Density in stabbing
    int dsty, maxDsty;
    
    // Counters
    int numberOfLimits, numOfInsertions;
    
    // Rotation matrices
    double Ropt[9], Raux[9], Rx[9], Ry[9];
    
    // Inclinations
    double xinc, yinc;
    
    // Stabbing vars. to control iterations.
    int i, offset;
    
    // Stabbing opt. val.
    double opt;
    
    int optIdx;
    
    // Lower bound for subproblem k.
    int lwbndK;
    
    // Feasible region limits.
    double uth, lth;
    
    // Array to be sorted for the stabbing algorithm
    struct LimitLabel *ll = new struct LimitLabel[4*numOfPts];
    
    // Vector indicating point matches to be tested.
    std::vector<bool> potentialOutlrs(numOfPts, true);
    
    //Euclidean distances
    double *Distance = (double *)mxMalloc(numOfPts*sizeof(double));
    
    double xk[3], yk[3];
    
    // Iterate to try to reject matches.
    for (int k=0; k<numOfPts; )
    {
        if (lwbnd>numOfPts)
        {
            break;
        }
        
        // Move to a potential outlier to reject
        while (k<numOfPts && !potentialOutlrs[k])
        {
            k++;
        }
        
        if (k==numOfPts) //out of the scope
        {
            break;
        }
        
        // if | xlen - ylen | > th , k is an outlier
        if( fabs(XLen[k]-YLen[k])>th )
        {
            mxAssert(numOfPts-1>=0, "out of scope");
            
            numOfPts--;
           // mxAssert(numOfPts>=lwbnd, "");
            
            if (k<numOfPts) //reorganise memory if k is not the last point
            {
                const int lstPt = 3*numOfPts;
                std::copy(X+lstPt,X+lstPt+3,X+3*k);
                std::copy(Y+lstPt,Y+lstPt+3,Y+3*k);
                XLen[k]=XLen[numOfPts];
                YLen[k]=YLen[numOfPts];
                H[k]=H[numOfPts];
                TH[k]=TH[numOfPts];
                potentialOutlrs[k]=potentialOutlrs[numOfPts];
            }
            
            continue;
        }
        
        
        if (XLen[k] <= DUMMY_PRECISION || YLen[k] <= DUMMY_PRECISION )
        {
            //inlier -> must not be removed
            k++;
            continue;
        }
        
        // Find Rx and Ry to align the visited point match with the N. Pole
        if (!rotationFromVector(X+k*3, Rx) || !rotationFromVector(Y+k*3, Ry))
        {
            k++;
            continue;
        }
        
        //eps k
        //double epsk = circleIntersectionAngle(XLen[k], YLen[k], th); //R, d, r
        double epsk = TH[k];
        if (epsk>MINTH)
        {
            k++;
            continue;
        }
        
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
            
            //ensure xi, yi match up to th
            
            if( fabs(XLen[i]-YLen[i])>th )
            {
                continue;
            }
            
            if ( XLen[i] <= DUMMY_PRECISION || YLen[i] <= DUMMY_PRECISION )
            {
                offset++;
                continue;
            }
            
            //xk = Rx * X[3*i]
            rotatePoint(xk, Rx, X+3*i);
            rotatePoint(yk, Ry, Y+3*i);
            
            double &x3 = xk[2];
            double &y3 = yk[2];
            
            if (y3>1)
            {
                y3=1.0;
            }
            
            mxAssert(fabs(y3)<=1, "z-coordinate out of range");
            
            yinc = acos(y3);
            
            mxAssert(yinc>=0 && yinc<=PI ,"incorrect inclination" );
            
            //Find eps i
            //double epsi = circleIntersectionAngle(XLen[i], YLen[i], th); //R, d, r
            double epsi = TH[i];
            
            if(epsi>MINTH)
            {
                offset++;
                continue;
            }
            
            // Make sure yinc is well defined (Non NAN values)
            if ( (1.0-x3 < DUMMY_PRECISION)  ) //x = North Pole
            {
                // if y touches the pole it also matches x
                if ( 1.0-y3 <= DUMMY_PRECISION || yinc <= epsk)
                {
                    offset++;
                }
                continue;
            }

            if ( x3+1.0 < DUMMY_PRECISION ) //x = South Pole
            {
                // if y touches the south it also matches x
                if ( y3+1.0 <= DUMMY_PRECISION || yinc >= PI-epsk)
                {
                    offset++;
                }
                continue;
            }
            
            xinc = acos(x3);

            // Make sure xinc is well defined (Non NAN values)
            
            // y = North Pole, South Pole
            if ( (1.0-y3 < DUMMY_PRECISION) || (y3+1.0 < DUMMY_PRECISION) )
            {
                offset++;
                continue;
            }
            
            //---------------------------------------------------------------
            // x and y are not in the poles
            //---------------------------------------------------------------
            mxAssert(xinc>0 && xinc<PI,"");
            mxAssert(yinc>0 && yinc<PI,"");
            
            //projections
            double &x1 = xk[0];
            double &x2 = xk[1];
            double &y1 = yk[0];
            double &y2 = yk[1];
            
            dot = (x1*y1+x2*y2)/sqrt((x1*x1+x2*x2)*(y1*y1+y2*y2));
            mxAssert(dot>=-1 && dot<=1,"");
            
            feasibleRegrionAngles(x3, xinc, dot, epsk, uth, lth);
            uth+=epsk; //add patch uncertainty
            lth+=epsk;
            
            // No solutin if x is out of the feasible region
            if(xinc>yinc)
            {
                if( (xinc-yinc) > lth+epsi)
                {
                    continue;
                }
            }
            else if( (yinc-xinc) > uth+epsi)
            {
                continue;
            }
            
            // Check Poles
            if ( ( (xinc-uth-epsk) <= 0 ) || ( (xinc+lth+epsk) >= PI ) ||
                 ( (yinc-epsk    ) <= 0 ) || ( (yinc+epsk    ) >= PI ) )
            {
                offset++;
                continue;
            }
            
            angdist = acos(dot);
            mxAssert(angdist>=0 && angdist<= PI,"");
            
            // Create segments in [0,2pi]
            theta = x1*y2-x2*y1 >=0 ?  angdist: TWOPI - angdist;
            
            if (!angularBoundingInterval(theta,xinc,yinc,epsk, epsi,thetaL,thetaR))
            {
                offset++;
                continue;
            }
            
            if (thetaL<=0 && thetaR>=TWOPI)
            {
                offset++;
                continue;
            }
            
            if (thetaL<0 || thetaR>TWOPI)
            {
                //compute two segments and check if them intersects
                if (thetaL<0 ) // l---[---r     ]
                {
                    segAEnd = thetaR;
                    segBSrt = TWOPI+thetaL;
                }
                else // [    l--]---r
                {
                    segAEnd = thetaR-TWOPI;
                    segBSrt = thetaL;
                }
                
                if (segBSrt<=segAEnd) // seg_a and seg_b intersect
                {
                    offset++;
                }
                else
                {
                    ll[numberOfLimits].value = 0;       ll[numberOfLimits++].label = 1;
                    ll[numberOfLimits].value = segAEnd; ll[numberOfLimits++].label =-1;
                    ll[numberOfLimits].value = segBSrt; ll[numberOfLimits++].label = 1;
                    ll[numberOfLimits].value = TWOPI;   ll[numberOfLimits++].label =-1;
                    numOfInsertions++;
                }
            }
            else //segment_l>0 && segment_r<TWOPI
            {
                ll[numberOfLimits].value = thetaL;  ll[numberOfLimits++].label = 1;
                ll[numberOfLimits].value = thetaR;  ll[numberOfLimits++].label =-1;
                numOfInsertions++;
            }

        } // End iterate over intervals
        
        if ( numOfInsertions+offset < lwbnd )
        {
            mxAssert(numOfPts-1>=0, "out of scope");

            numOfPts--;

            if (k<numOfPts) //reorganise memory if k is not the last point
            {
                const int lstPtPtr = 3*numOfPts;
                std::copy(X+lstPtPtr,X+lstPtPtr+3,X+3*k);
                std::copy(Y+lstPtPtr,Y+lstPtPtr+3,Y+3*k);
                XLen[k]=XLen[numOfPts];
                YLen[k]=YLen[numOfPts];
                H[k]=H[numOfPts];
                TH[k]=TH[numOfPts];
                potentialOutlrs[k]=potentialOutlrs[numOfPts];
            }
            
            //use lower bound?
          //  mxAssert(numOfPts>=lwbnd, "");
            
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
        
        lwbndK = eval(Ropt,X,Y,XLen,YLen,Rx,Ry,numOfPts,th,lwbnd);
        
        // Update lower bound
        if(lwbndK > lwbnd)
        {
            lwbnd = lwbndK;
            
            //Compute distances D from position k+1
            residuals(k+1,Ropt,X,Y,XLen,YLen,Rx,Ry,Distance,numOfPts);
            
            
            // Update array of potential outlrs from position k+1
            for (i=k+1; i<numOfPts; i++)
            {
                potentialOutlrs[i] = Distance[i]>sqrdTh;
            }
            
            // Update output rotation matrix: Rout = Ry.transpose()*Ropt*Rx;
            matrix3Multiply(Raux, Ropt, Rx);
            matrix3Multiply(Rout, transpose(Ry), Raux);
        }
        
        //------------------------------------------------
        // Check upper bound estimation
        //------------------------------------------------
        if (maxDsty < lwbnd)
        {
            numOfPts--;

            if (k<numOfPts) //reorganise memory if k is not the last point
            {
                const int lstPtPtr = 3*numOfPts;
                std::copy(X+lstPtPtr,X+lstPtPtr+3,X+3*k);
                std::copy(Y+lstPtPtr,Y+lstPtPtr+3,Y+3*k);
                XLen[k]=XLen[numOfPts];
                YLen[k]=YLen[numOfPts];
                H[k]=H[numOfPts];
                TH[k]=TH[numOfPts];
                potentialOutlrs[k] = potentialOutlrs[numOfPts];
            }
            
          //  mxAssert(numOfPts>=lwbnd, "");
            continue;
        }
        
        // Move to the next correspondence to be tested
        k++;
    } // End iteration over correspondences
    
    delete []ll;
    mxFree(Distance);
    
    outLwbnd = lwbnd;
   
    return numOfPts;
}

// return new size of H
int _gore_l2_rep(size_t *H, double *X, double *Y, double *XLen, double *YLen, int numOfPts,
                   double th,double *TH,int lwbnd, double *Rout, int &outLwbnd)
{
    mxAssert(numOfPts>0,"invalid number of points");
    mxAssert(lwbnd>=0,"invalid lower bound");
    mxAssert (th>0,"invalid threshold");
    
    int n = _gore_l2(H,X,Y,XLen,YLen,numOfPts,th,TH,lwbnd,Rout,outLwbnd);
    mxAssert(outLwbnd>=lwbnd, "");
    
    int bestLwbnd = outLwbnd;
    
    double R[9];
    
    while ( n < numOfPts)
    {
        numOfPts = n;
        
        n = _gore_l2(H,X,Y,XLen,YLen,numOfPts,th,TH,bestLwbnd,R,outLwbnd);
        
        //Update the best solution
        if (outLwbnd > bestLwbnd)
        {
            bestLwbnd = outLwbnd;
            std::copy(R,R+9,Rout);
        }
    }
    mxAssert(n==numOfPts, "");
    
    outLwbnd = bestLwbnd;
    
    return numOfPts;
}


inline
void normalisecols(const double *X, int n, double *out, double *len)
{
    mxAssert(n>=0, "invalid number of points");
    int i,j;
    for(j=0; j<n; j++) /* Compute a matrix with normalized columns */
    {
        len[j] = 0.0;
        for(i=0; i<3; i++)
        {
            len[j] += (X[i + 3*j])*(X[i + 3*j]);
        }
        len[j] = sqrt(len[j]);
        for(i=0; i<3; i++)
        {
            out[i + 3*j] = X[i + 3*j]/len[j];
        }
    }
}


int reg::gore::gore3_l2(size_t *H, const double *X_, const double *Y_,
                        int numOfPts, double th, int lwbnd,
                        bool repFlag, double *R, int &outLwbnd)
{
    mxAssert(numOfPts>=0 , "invalid number of points");
    mxAssert(lwbnd>=0 , "invalid input lower bound");
    mxAssert(th>0 , "invalid threshold");
    
    //Normalise points
    double *X    = (double *)mxMalloc(3*numOfPts*sizeof(double));
    double *Y    = (double *)mxMalloc(3*numOfPts*sizeof(double));
    double *XLen = (double *)mxMalloc(numOfPts*sizeof(double));
    double *YLen = (double *)mxMalloc(numOfPts*sizeof(double));
    double *TH = (double *)mxMalloc(numOfPts*sizeof(double));
    
    normalisecols(X_, numOfPts, X, XLen);
    normalisecols(Y_, numOfPts, Y, YLen);
 

    for(int i=0;i<numOfPts;i++)
    {
        TH[i] = circleIntersectionAngle(XLen[i], YLen[i], th); //R, d, r
    }
    
    int n;
    if (!repFlag)
    {
        n = _gore_l2(H,X,Y,XLen,YLen,numOfPts,th,TH,lwbnd,R,outLwbnd);
    }
    else
    {
        n = _gore_l2_rep(H,X,Y,XLen,YLen,numOfPts,th,TH,lwbnd,R,outLwbnd);
    }
 
    mxAssert(numOfPts-n>=0, "");
    mxAssert(outLwbnd>=lwbnd, "");
    
    mxFree(X); mxFree(XLen);
    mxFree(Y); mxFree(YLen);
    mxFree(TH);

    return n;
}

