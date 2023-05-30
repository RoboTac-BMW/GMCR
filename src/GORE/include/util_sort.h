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

#include <stdlib.h>
#include <string.h>
#include "reg_common.h"


#ifndef REG_SORT_H
#define REG_SORT_H


namespace reg
{
namespace util
{

//TODO: use a template
struct value_index
{
    double value;
    unsigned int index;
};

struct value_index_uint
{
    unsigned int value;
    unsigned int index;
};


int compare_value_index(const void * a, const void * b);
int compare_value_index_uint(const void * a, const void * b);

unsigned int* sort_index(double *a, unsigned int len);
unsigned int* sort_index(unsigned int *a, unsigned int len);

void sorted_by_index(double *a, unsigned int* idx, unsigned int len);
void sorted_by_index(int *a, unsigned int* idx, unsigned int len);


/*
 * Sort an array according to indexes in idx. An aux array is given.
 */
void sorted_by_index2(double *a, unsigned int* idx, unsigned int len, double *tmp);


} // End namespace util
} // End namespace reg

#endif
