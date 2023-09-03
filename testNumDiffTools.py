import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian

def fun(x, a):
    return (x[0]-1)**2 + (x[1] - a) ** 2

def fun_der(x, a):
    return Jacobian(lambda x: fun(x, a))(x).ravel()

def fun_hess(x, a):
    return Hessian(lambda x: fun(x, a))(x)


x0 = np.array([2, 0])
a = 2.5

res = minimize(fun, x0, args=(a,), method='dogleg', jac=fun_der, hess= fun_hess)

print(res)

## 

A = np.random.rand(5,3)
b = np.random.rand(5)

fun = lambda x: np.dot(x, A.T) - b

x = np.random.rand(3)

jac = Jacobian(fun)(x)

print(np.allclose(jac - A, 0))

