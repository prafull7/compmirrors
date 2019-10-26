import numpy as np
from scipy.sparse.linalg import svds

# computes the singular value decomposition of the input video for k values, and returns the U, Sigma and V matrices
def video_svdbasis(Z, k=96):
    print('computing singular vectors')

    s = Z.shape

    # reshape into [t, ij], and temporarily go to doubles for accuracy
    Z = np.reshape(Z, [s[0], s[1]*s[2]]).astype(np.float64)

    # compute truncated SVD
    U, S, V = svds(Z, k=k)

    U = np.flip(U, axis=1).astype(np.float32)   # [t, sv]
    S = np.flip(S).astype(np.float32)
    V = np.flip(V, axis=0).astype(np.float32)   # [sv, ij]
    return U, S, V
