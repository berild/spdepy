from .matrix import Matrix
import numpy as np
from scipy import sparse

def diags(vec: np.ndarray) -> Matrix:
    """Create a sparse diagonal matrix from a vector of the diagonal entries
        Constructor for the sparse matrix class.
        
    :param vec: vector of diagonal entries
    :type vec: np.ndarray
    :return: Matrix class
    :rtype: Matrix
    """
    return(Matrix(sparse.diags(vec).tocsc()))

def eye(n: int) -> Matrix:
    """Create a sparse identity matrix of size n

    :param n: size of identity matrix
    :type n: int
    :return: Matrix class
    :rtype: Matrix
    """
    return(Matrix(sparse.eye(n).tocsc()))

def init(*args, **kwargs) -> Matrix:
    """Create a sparse matrix using the same constructor as sparse.csc_matrix
    
    :param args: arguments to sparse.csc_matrix
    :type args: tuple
    :param kwargs: keyword arguments to sparse.csc_matrix
    :type kwargs: dict
    :return: Matrix class
    :rtype: Matrix
    """
    if len(args) == 1:
        return(Matrix(sparse.csc_matrix(args[0])))
    elif len(args) ==2 and kwargs.get("shape") is not None:
        print(kwargs.get("shape"))
        return(Matrix(sparse.csc_matrix(args, shape=kwargs.get("shape"))))
    
    