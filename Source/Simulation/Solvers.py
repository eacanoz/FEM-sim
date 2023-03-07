import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Source.core import Model

def solve(model:Model):

    Nfeval = 1

    model.constructProblem()

    x0 = np.ones(model.mesh.getNoN())*298 ## Change!!!
    xi_1 = x0

    x, exitCode = sc.sparse.linalg.bicg(model.A)



def callbackF(xi):

    global Nfeval, xi_1


def pyPARDISO(model:Model):
    
    # A = model.A
    b = model.b.todense()

    A = sp.csr_matrix([[4, 0, 0, 0], [0, 2, 0, 1], [0, 0, 3, 0], [0, 1, 0, 2]])

    lu = spla.splu(A)

    y = lu.solve(b)
    x = lu.solve(y)
    
    spla.spsolve(A, b, use_umfpack = False)
    
    pass


