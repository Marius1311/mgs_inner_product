from typing import Optional
import numpy as np
from scipy.linalg import eigh



def mgs_inner_prod(Z: np.ndarray, A: Optional[np.ndarray] = None, check_pd: bool = True):
    """
    Modified Gram-Schmidt orthogonalisation with custom inner product defined by A

    This implements Algorithm 2 of Imakura & Yamamoto, 2019.
    https://doi.org/10.1007/s13160-019-00356-4

    Acts on the columns of the (m x n) matrix Z, where we assume m >= n. Further, we
    require A to be symmetric positive definite. This is one possiblitiy to compute the
    generalized QR decompsotion of a matrix with respect to a non-standard inner product.

    Parameters
    --------
    Z
        (m x n) matrix to be orthogonalised with respect to A

    A
        (m x m) matrix which must define an inner product, i.e. it
        must be symmetric positive definite. If None, identify matrix
        is assumed
    check_pd
        checks whether A is positive definite. This can be turned off -
        it's usually the most expensive computation performed. 

    Returns
    --------
    Q
        (m x n) matrix whose columns are orthogonal with respect to A
    R
        (n x n) upper triangular matrix

    It holds: Z = QR
    """

    # initialise
    (m, n) = Z.shape
    if A is None:
        A = np.eye(m)
    (l, k) = A.shape
    Q, R, P = np.zeros((m, n)), np.zeros((n, n)), np.zeros((m, n))

    # check input parameters
    assert (l == k), "A is not rectangular"
    assert (m == l), "Shape mismatch"
    assert (np.allclose(A, A.T)), "A is not symmetric"
    if check_pd:
        assert (all(eigh(A, eigvals_only=True) > 0)), "A is not positive definite"

    # orthogonalise
    for j in range(n):
        z = Z[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(P[:, i], z)
            z -= R[i, j] * Q[:, i]
        x = np.dot(A, z)
        R[j, j] = np.sqrt(np.dot(z, x))
        Q[:, j] = z / R[j, j]
        P[:, j] = x / R[j, j]

    # check that Q fulfills the orthogonality relation wrt A
    assert (np.allclose(Q.T @ A @ Q, np.eye(n))), "Q does not fulfill orthogonality wrt A"

    # check that Z = QR holds
    assert (np.allclose(Z, Q @ R)), "Z = QR does not hold"

    return Q, R