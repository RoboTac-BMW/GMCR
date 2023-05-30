import numpy as np
import networkx as nx


def clipper_get_max_clique(M):

    pass

def create_affinity_matrix(pointcloud_pair):
    """
    Will return the affinity matrix of the consistency graph
    :param pointcloud_pair: Instance of PointCloudPair class
    :return: Affinity matrix as nump.ndarray
    """
    max_node_num = max(pointcloud_pair.source.points.shape[0], pointcloud_pair.source.points.shape[0])

    pass


def find_dense_cluster(M, C):
    """

    :param M: Affinity matrix of the graph
    :param C: Constraint matrix, where C(i,j)=0, if the connection should be ignored
    :return:
    """

    # Parameters
    max_iters = 1000 # Num of overall iterations with updating Md
    max_inner_iters = 200 # Num iterations on gradient updates without updating Md
    max_ls_iters = 99 # Num iterations for line search with backtracking
    eps = 1e-9 # Value for idxD
    beta = 0.25 # Value for scaling alpha in line search
    tol_u = 1e-8 # Value for change in norm of u when to stop inner optimization
    tol_F = 1e-9 # Value for change in F when to stop optimization

    N = M.shape[0]

    # Initialize u uniform
    u0 = np.random.uniform(0, 1, size=N)

    # Zero out values in M that correspond to an active constraint
    M = M*C

    # Create the binary complement of the constraint matrix (1->0 and 0->1)
    Cb = np.ones_like(C) - C

    # Perform one step of power iteration to get a good scaling of u
    u = M @ u0
    u /= np.linalg.norm(u)

    # Initializing d, however its not clear what the intuition behind that is
    d = 0 # this is the case when no constraints are active
    Cbu = Cb @ u
    idxD = np.logical_and((Cbu > eps), (u > eps))

    if idxD.sum() > 0:
        Mu = M@u
        num = np.where(idxD, Mu, np.finfo('float32').max)
        den = np.where(idxD, Cbu, 1)
        d = (num / den).min()

    # Create Homotopy (matrix with -d where the constraints are active, i.e. there is no edge)
    Md = M - d * Cb

    for i in range(max_iters):

        # calculate current objective value
        F = u @ Md @ u

        for j in range(max_inner_iters):

            # Calculate gradient of objective function
            gradF = Md @ u

            # ToDo: projection onto the sphere (is skipped on github)
            # Run line search with backtracking to find the optimal step size alpha
            alpha = 1 # starting value for alpha
            unew = None
            Fnew = None

            for i in range(max_ls_iters):
                unew = u + alpha * gradF
                unew = np.maximum(unew, 0) # project onto positive orthant
                unew /= np.linalg.norm(unew) # project onto unit-sphere
                Fnew = unew @ Md @ unew # new objective value

                deltaF = Fnew - F # Change in objective value

                # if < 0 means that the objective value decreased (worse than before since maximization)
                if deltaF < 0:
                    alpha = alpha * beta

                # This means we have found a u with a better objective value
                else:
                    break

            # Calculate change in u
            deltau = np.linalg.norm(unew - u)

            # actually performing the update step of u and getting new objective value
            u = unew
            F = Fnew

            # Stop optimization if u has converged
            if deltau < tol_u or deltaF < tol_F:
                break

        # Decrease value of d
        Cbu = Cb @ u
        idxD = np.logical_and((Cbu > eps), (u > eps))

        # ToDo: Probably here lies the problem
        if idxD.sum() > 0:
            Mu = M@u
            num = np.where(idxD, Mu, np.finfo('float32').max)
            den = np.where(idxD, Cbu, 1)
            deltad = np.absolute(num / den).min() # absolute is different from definition above, not clear why

            d += deltad
            Md = M - d * Cb
        else:
            break

    # Generate the result
    # Estimate cluster size using largest eigenvalue
    omega = (u @ Md @ u).round().astype(int)

    # Get indices of largest [omega]-vaues
    cluster_inds = np.flip(np.argsort(u))[:omega]

    return cluster_inds


def get_binary_constraint_matrix(M):

    C = np.ones_like(M)
    C[M == 0] = 0

    return C


if __name__ == "__main__":
    pass