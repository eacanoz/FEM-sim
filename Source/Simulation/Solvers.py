import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scla

# from Source.core import Model

Nfeval = 1
xi_1 = []

def constructProblem(model):
    A = model.assembleGlobalMatrix()
    b = model.assembleGlobalVector()

    A, b = applyBC(model, A, b)

    return A.tocsc(), b

def applyBC(model, A, b):

    print('Applying Boundary Conditions')

    for node in model._mesh.NL:
        if node.BC is not None and node.BC['type'] == 'Dirichlet':
            A[node.id, :] = 0
            A[node.id, node.id] = 1

            b[node.id] = node.BC['value']

    return A, b


def solve(model, x0):

    global xi_1

    xi_1.append(x0)

    #model.constructProblem()

    A, b = constructProblem(model)

    #x0 = np.ones(model.mesh.getNoN())*298 ## Change!!!
    #
    print('')

    print('{0:4s}   {1:9s}'.format('Iter', 'error'))
    x, exitCode = spla.bicgstab(A, b, x0=x0, callback=callbackF)

    print('{0:4d}   {1:3.14f}'.format(Nfeval, scla.norm((x-xi_1[Nfeval-1]))))
    print('')
    

    return x, exitCode


def callbackF(xi):

    global Nfeval
    global xi_1

    itError = scla.norm((xi-xi_1[Nfeval-1]))

    print('{0:4d}   {1:3.14f}'.format(Nfeval, itError))
    xi_1.append(xi.copy()) 
    Nfeval += 1



# def pyPARDISO(model:Model):
    
#     # A = model.A
#     b = model.b.todense()

#     A = sp.csr_matrix([[4, 0, 0, 0], [0, 2, 0, 1], [0, 0, 3, 0], [0, 1, 0, 2]])

#     lu = spla.splu(A)

#     y = lu.solve(b)
#     x = lu.solve(y)
    
#     spla.spsolve(A, b, use_umfpack = False)
    
#     pass


