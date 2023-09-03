# Functions
import numpy as np
import math

def assign_BCs(NL, ENL):
    PD = np.size(NL, 1)  # Problem Dimension
    NoN = np.size(NL, 0)  # Number of nodes

    DOFs, DOCs = 0, 0

    for i in range(0, NoN):
        for j in range(0, PD):

            if ENL[i, PD + j] == -1:
                DOCs -= 1
                ENL[i, 2 * PD + j] = DOCs

            else:
                DOFs += 1
                ENL[i, 2 * PD + j] = DOFs

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, 2 * PD + j] < 0:
                ENL[i, 3 * PD + j] = abs(ENL[i, 2 * PD + j]) + DOFs
            else:
                ENL[i, 3 * PD + j] = abs(ENL[i, 2 * PD + j])

    DOCs = abs(DOCs)

    return ENL, DOFs, DOCs

def assemble_stiffness(ENL, EL, NL, E, A):
    NoE = np.size(EL, 0)    # Number of elements
    NPE = np.size(EL, 1)    # Nodes por Element
    PD = np.size(NL, 1)  # Problem Dimension
    NoN = np.size(NL, 0)  # Number of nodes

    K = np.zeros([NoN*PD, NoN*PD])

    for i in range(0, NoE):
        nl = EL[i,0:NPE]
        k = element_stiffness(nl, ENL, E, A) # Element stiffness matrix

        for r in range(0, NPE):
            for p in range(0, PD):
                for q in range(0, NPE):
                    for s in range(0, PD):
                        row = ENL[nl[r]-1, p+3*PD]
                        column = ENL[nl[q]-1, s+3*PD]
                        value = k[r*PD+p, q*PD+s]
                        K[int(row)-1, int(column)-1] = K[int(row)-1, int(column)-1] + value

    return K


def element_stiffness(nl, ENL, E, A):
    X1 = ENL[nl[0] - 1, 0]
    Y1 = ENL[nl[0] - 1, 1]

    X2 = ENL[nl[1] - 1, 0]
    Y2 = ENL[nl[1] - 1, 1]

    L = math.sqrt((X1-X2)**2 + (Y1-Y2)**2)      # Element length

    C = (X2-X1)/L   # Cosine
    S = (Y2-Y1)/L   # Sine

    k = (E*A)/L * np.array([[C**2, C*S, -C**2, -C*S],
                            [C*S, S**2, -C*S, -S**2],
                            [-C**2, -C*S, C**2, C*S],
                            [-C*S, -S**2, C*S, S**2]])

    return k

def assemble_forces(ENL, NL):
    PD = np.size(NL, 1)  # Problem Dimension
    NoN = np.size(NL, 0)  # Number of nodes
    DOF = 0

    Fp = []

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == 1:
                DOF += 1
                Fp.append(ENL[i, 5*PD+j])

    Fp = np.vstack([Fp]).reshape(-1, 1)

    return Fp

def assemble_displacements(ENL, NL):
    PD = np.size(NL, 1)  # Problem Dimension
    NoN = np.size(NL, 0)  # Number of nodes
    DOC = 0

    Up = []

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD + j] == -1:
                DOC += 1
                Up.append(ENL[i, 4 * PD + j])

    Up = np.vstack([Up]).reshape(-1, 1)

    return Up

def update_nodes(ENL, U_u, NL, Fu):

    PD = np.size(NL, 1)  # Problem Dimension
    NoN = np.size(NL, 0)  # Number of nodes

    DOFs, DOCs = 0, 0

    for i in range(0, NoN):
        for j in range(0, PD):

            if ENL[i, PD + j] == 1:
                DOFs += 1
                ENL[i, 4 * PD + j] = U_u[DOFs-1]

            else:
                DOCs += 1
                ENL[i, 5 * PD + j] = Fu[DOCs-1]

    return ENL