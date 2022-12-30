import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    x1_a = np.array([[0,-1*x1[2],x1[1]],[x1[2],0,-1*x1[0]],[-1*x1[1],x1[0],0]])
    x2_a = np.array([[0,-1*x2[2],x2[1]],[x2[2],0,-1*x2[0]],[-1*x2[1],x2[0],0]])
    
    x1_c = np.dot(x1_a,P1)
    x2_c = np.dot(x2_a,P2)
    A = np.concatenate((x1_c,x2_c))
    #did not get implementation with np.linalg.eig to work -> this works
    u,s,w=np.linalg.svd(A) 
    smallest=np.argmin(s)
    X = w[smallest]


    ##-your-code-ends-here-##

    #X = np.array([0, 0, 0, 1])  # remove me
    
    return X
