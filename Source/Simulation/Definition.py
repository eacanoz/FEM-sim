## Problem Definition

# General form of Equations

# Examples

# Continuity equation
Cont_Eq = 'ddt(rho) + div(rho, v_) == 0'

# Momentum equation
Mom_Eq = 'ddt(rho,v_) + div(rho, v_, v_) == lap(miu, v_) - grad(p) + F '

# Steady state conduction

SS_Cond_Eq = 'lap(k,T) = 0'

## Definitions

# ddt -> Temporal derivative
# div -> Divergency operator
# grad -> Gradient Operator
# lap -> Laplacian operator

# The idea is to take the defined equation and transform it into the equivalent Matrix-Vector Weak Form