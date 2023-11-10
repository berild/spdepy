from scipy import sparse
import numpy as np
from sksparse.cholmod import cholesky

class Matrix:
    """ _summary_

    Returns
    -------
    _type_
        _description_
    """
    cholExists = False

    
    def __init__(self, mat: sparse.csc_matrix) -> None:
        """__init__ _summary_

        Parameters
        ----------
        mat : sparse.csc_matrix
            _description_
        """
        
        self.mat = mat
    
    @property
    def T(self) -> sparse.csr_matrix:
        """Tanspose of a sparse matrix

        Returns
        -------
        sparse.csr_matrix
            transpose of sparse matrix
        """
        return self.mat.T

    @property
    def shape(self) -> np.ndarray:
        """shape of sparse matrix

        Returns
        -------
        np.ndarray
            [nrows, ncols] of sparse matrix
        """
        return self.mat.shape
    
    @property
    def size(self) -> int:
        """size of sparse matrix

        Returns
        -------
        int
            size of sparse matrix
        """
        return self.mat.T
    
    @property
    def H(self) -> sparse.csr_matrix:
        
        """
        Hermitean transpose
        
        Returns
        -------
        sparse.csr_matrix
            Hermitean transpose of sparse matrix
        """
        return self.mat.H
    
    @property
    def A(self) -> np.ndarray:
        """
        Densify matrix
        
        Returns
        -------
        np.ndarray
            dense representation of sparse matrix
        """
        return self.mat.A
    
    def reshape(self, *args, **kwargs):
        """
        Gives a new shape to a sparse array/matrix without changing its data.

        :param shape:  length-2 tuple of ints
        :type shape: tuple
        :param order:  {'C', 'F'}, optional
            Read the elements using this index order. 'C' means to read and
            write the elements using C-like index order; e.g., read entire first
            row, then second row, etc. 'F' means to read and write the elements
            using Fortran-like index order; e.g., read entire first column, then
            second column, etc.
        :param copy: Indicates whether or not attributes of self should be copied
            whenever possible. The degree to which attributes are copied varies
            depending on the type of sparse array being used.
        :type copy: bool, optional
        :return: reshaped sparse matrix class
        :rtype: Matrix
        """
        return Matrix(self.mat.reshape(args, kwargs))    
            
    
    def toarray(self) -> np.ndarray:
        """
        Densify matrix

        :return: dense representation of sparse matrix
        :rtype: np.ndarray
        """
        return self.mat.toarray()
    
    def trace(self) -> float:
        """
        Trace of a scipy.sparse.csc_matrix
        
        :return: Trace of the matrix
        :rtype: float
        """
        return self.mat.trace()
    
    def transpose(self) -> sparse.csr_matrix:
        """
        Transpose of matrix

        :return: transpose of sparse matrix
        :rtype: sparse.csr_matrix
        """
        return self.mat.transpose()
    
    def diagonal(self,k=0) -> np.ndarray:
        """diagonal _summary_

        :param k: _description_, defaults to 0
        :type k: int, optional
        :return: _description_
        :rtype: np.ndarray
        """
        return self.mat.diagonal( k=k )
    
    def pinv(self, method = "qinv") -> sparse.csc_matrix:
        """Partial inverse of a sparse matrix

        :param method: _description_, defaults to "qinv"
        :type method: str, optional
        :return: _description_
        :rtype: sparse.csc_matrix
        """
        return self.mat
    
    def _chol(self) -> None:
        """Cholesky decomposition of sparse matrix
        Uses the sksparse.cholmod.cholesky function to decompose the sparse matrix
        
        """
        self.fac = cholesky(self.mat)
        self.cholExists = True
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """solve the linear equation set A@ x == b for the uknnown x for a positive definite matrix A.

        Parameters
        ----------
        b : np.ndarray
            Input array for the right hand side

        Returns
        -------
        np.ndarray
            the solution array x
        """
        if not self.cholExists:
            self.fac = self._chol(self.mat)
        return self.fac.solve_A(b)
    
    def solve_L(self, b: np.ndarray) -> np.ndarray:
        """solve the linear system L@ x == b for the uknnown x for a lower triangular matrix L.
        L is the cholesky decomposition of a positive definite matrix A.

        Parameters
        ----------
        b : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        if not self.cholExists:
            self.fac = self._chol(self.mat)
        return self.fac.solve_L(b)
    
    def update(self, mat: sparse.csc_matrix) -> None:
        self.mat = mat
        if self.cholExists:
            self.fac.update_inline(mat)
    
    def __add__(self, mat: sparse.csc_matrix) -> sparse.csc_matrix:
        return self.mat + mat
    
    def __radd__(self, mat: sparse.csc_matrix) -> sparse.csc_matrix:
        return mat + self.mat
    
    def __sub__(self, mat: sparse.csc_matrix) -> sparse.csc_matrix:
        return self.mat - mat
    
    def __rsub__(self, mat: sparse.csc_matrix) -> sparse.csc_matrix:
        return mat - self.mat
    
    def __mul__(self, vec):
        return self.mat * vec

    def __rmul__(self, vec):
        return vec * self.mat

    def __matmul__(self, mat):
        return Matrix(self.mat@mat)
    
    def __rmatmul__(self, mat):
        return Matrix(self.mat@mat)
    
    def __call__(self) -> None:
        # Not implemented yet
        pass
    
    def __eq__(self, __value: object) -> bool:
        # Not implemented yet
        pass
    
    def __repr__(self) -> str:
        pass
    
    def __str__(self) -> str:
        pass