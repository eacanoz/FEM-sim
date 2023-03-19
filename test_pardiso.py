import pypardiso

import numpy as np
import scipy.sparse as sp

A = sp.rand(10, 10, density=0.5, format='csr')

# print(A)

b = np.random.rand(10)

x = pypardiso.spsolve(A, b)

print(x)