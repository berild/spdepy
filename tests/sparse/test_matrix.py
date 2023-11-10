import numpy as np
from spdepy import sparse
from scipy import sparse as old_sparse



# construction tests    
def test_init_diag():
    sp = sparse.diag([1, 2, 3])
    assert np.array_equal(sp.toarray(), np.diag([1, 2, 3]))
    
def test_init_eye():
    sp = sparse.eye(10)
    assert np.array_equal(sp.toarray(), np.eye(10))
    
def test_init_sparse_matrix():
    arr = np.array([[1, 2], [3, 4]])
    osp = old_sparse(arr)
    sp = sparse(osp)
    assert np.array_equal(sp.toarray(), arr)

def test_init_matrix():
    arr = np.array([[1, 2], [3, 4]])
    sp = sparse(arr)
    assert np.array_equal(sp.toarray(), arr)
    
    
# operation tests
def test_transpose():
    arr = np.array([[1, 2], [3, 4]])
    sp = sparse(arr)
    assert np.array_equal(sp.transpose().toarray(), arr.T)
    assert np.array_equal(sp.T.toarray(), arr.T)

def test_equal():
    arr = np.array([[1, 2], [3, 4]])
    sp = sparse(arr)
    assert np.array_equal(sp, arr)
    
def test_scalar_multiply():
    arr = np.array([[1, 2], [3, 4]])
    sp = sparse(arr)
    assert np.array_equal(sp * 2, arr * 2)
    
def test_array_multiply():
    arr = np.array([[1, 2], [3, 4]])
    sp = sparse(arr)
    assert np.array_equal(sp * arr, arr * arr)
    
def test_matrix_multiply():
    arr = np.array([[1, 2], [3, 4]])
    sp = sparse(arr)
    assert np.array_equal(sp @ sp, arr @ arr)
    
def test_addition():
    arr = np.array([[1, 2], [3, 4]])
    sp = sparse(arr)
    assert np.array_equal(sp + sp, arr + arr)
    
def test_subtraction():
    arr = np.array([[1, 2], [3, 4]])
    sp = sparse(arr)
    assert np.array_equal(sp - sp, arr - arr)
