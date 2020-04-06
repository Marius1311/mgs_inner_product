# Modified Gram-Schmidt with non-standard inner product
This implements Algorithm 2 of Imakura & Yamamoto, 2019, which is a high-accuracy implementation of a modified Gram-Schmidt process with non-standard inner product. Given an (m x n) matrix Z with m >= n and a symmetric positive definite (m x m) matrix A defining an inner product, the algorithm returns a matrix Q and a matrix R such that Z = QR, where Q^T A Q = I and R is upper triangular. 

## References
Imakura, A., Yamamoto, Y. Efficient implementations of the modified Gram–Schmidt orthogonalization with a non-standard inner product. Japan J. Indust. Appl. Math. 36, 619–641 (2019). https://doi.org/10.1007/s13160-019-00356-4
