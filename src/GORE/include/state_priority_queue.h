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

#ifndef REG_STATE_PRIORITY_QUEUE_H_
#define REG_STATE_PRIORITY_QUEUE_H_

#include "state.h"
#include <cstddef> //NULL

namespace reg {
namespace search {


template <class SSR, typename Scalar=int >
class StatePriorityQueue
{
public:
    enum OptProblem{MINIMISATION, MAXIMISATION};

private:

    class Node
    {
    public:
        SearchState<SSR, Scalar> *state;
        Node *left, *right;

        Node(): state(NULL), left(NULL), right(NULL) {}
        Node(SearchState<SSR, Scalar> *state):state(state), left(NULL), right(NULL){}
        ~Node() {if (state!=NULL) delete state;}
    };

    const OptProblem optProblem;
    Node *head, *tail;
    unsigned int m_size;

public:
    StatePriorityQueue(OptProblem op=MAXIMISATION);
    ~StatePriorityQueue();

    SearchState<SSR, Scalar> *pop();
    void push(SearchState<SSR, Scalar> *state);

    /**
     * @brief Remove and free states with upper bound lower or equal to lwbnd.
     * @param lwbnd Known lower bound.
     */
    void prune(int curbest);

    unsigned int size() const;
};


} // End namespace search
} // End namespace reg

#include "state_priority_queue.hpp"

#endif
