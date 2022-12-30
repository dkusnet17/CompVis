import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def estimateF(x1, x2):
    """
    :param x1: Points from image 1, with shape (coordinates, point_id)
    :param x2: Points from image 2, with shape (coordinates, point_id)
    :return F: Estimated fundamental matrix
    """

    # Use x1 and x2 to construct the equation for homogeneous linear system
    ##-your-code-starts-here-##
    n = x1.shape[1]   
    u1 = x1[0, :]
    v1 = x1[1, :]
    u2 = x2[0, :]
    v2 = x2[1, :]
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [u2[i]*u1[i], u2[i]*v1[i], u2[i],
                v2[i]*u1[i], v2[i]*v1[i], v2[i], 
                u1[i], v1[i], 1 ]
    ##-your-code-ends-here-##

    # Use SVD to find the solution for this homogeneous linear system by
    # extracting the row from V corresponding to the smallest singular value.
    ##-your-code-starts-here-##
    S, V = linalg.svd(A)   #,U
    index = np.argmin(S)
    v_min = V[index, :]
    ##-your-code-ends-here-##
    F = np.reshape(v_min, (3, 3))  # reshape to acquire Fundamental matrix F
    #F = np.ones((3, 3))  # remove me and uncomment the above

    # Enforce constraint that fundamental matrix has rank 2 by performing
    # SVD and then reconstructing with only the two largest singular values
    # Reconstruction is done with u @ s @ vh where s is the singular values
    # in a diagonal form.
    ##-your-code-starts-here-##
    u, s, vh = linalg.svd(F)
    idx = np.argmin(s)
    s[idx] = 0
    F = u @ s @ vh  # np.diag(s)
    ##-your-code-ends-here-##
    #print(F)
    return F
    