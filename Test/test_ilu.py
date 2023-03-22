import numpy as np
from scipy import sparse
from scipy.sparse import linalg

A = np.array([[ 0.4445,  0.4444, -0.2222],
              [ 0.4444,  0.4445, -0.2222],
              [-0.2222, -0.2222,  0.1112]])
sA = sparse.csc_matrix(A)

b = np.array([[ 0.6667], 
              [ 0.6667], 
              [-0.3332]])

sA_iLU = sparse.linalg.spilu(sA)
M = sparse.linalg.LinearOperator(sA.shape, sA_iLU.solve)

x, info = sparse.linalg.gmres(A,b,M=M)

print(x)