import numpy as np
from clean_value import clean_value

# a function that finds the "square root" of a PSD matrix Q, i.e. the matrix B s.t. B^T*B=Q

def matrix_sqrt(Q):
    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    # Clamp small negative eigenvalues to 0
    eigenvalues = clean_value(np.maximum(eigenvalues, 0))
    # Construct sqrt of the diagonal matrix
    sqrt_Lambda = (np.diag(np.sqrt(eigenvalues)))

    # Construct B = U sqrt(Lambda)
    B = eigenvectors @ sqrt_Lambda

    return B