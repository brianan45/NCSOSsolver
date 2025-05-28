import numpy as np

# a function that finds the "square root" of a PSD matrix Q, i.e. the matrix B s.t. B*B^T=Q

def matrix_sqrt(Q):
    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    
    # Clamp small negative eigenvalues to 0
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Construct sqrt of the diagonal matrix
    sqrt_Lambda = np.diag(np.sqrt(eigenvalues))
    U = eigenvectors

    # Construct B = U sqrt(Lambda)
    B = U @ sqrt_Lambda

    return B