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

#ifndef REG_BINARYTREE_H
#define REG_BINARYTREE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"


namespace reg
{
namespace binarytree {


/* Definition for interval.*/
typedef struct interval
{
    double lw;
    double up;
} interval;

/* Definitions for binary search tree.*/
typedef struct payload
{
    double val;
    int order;
} payload;

typedef struct treeNode
{
    payload data;
    struct treeNode *left;
    struct treeNode *right;
} treeNode;

treeNode *Insert(treeNode*, payload, treeNode*);
void free_Binarytree(treeNode *node);
int queryLower(treeNode*,double,treeNode*);
int queryUpper(treeNode*,double,treeNode*);
double queryMiddle(treeNode *,double,treeNode *);
void PrintInorder(treeNode*);
int size_Binarytree(treeNode*);
int count_pointers_Binarytree(treeNode*);

} // End namespace binarytree
} // End namespace reg

#endif
