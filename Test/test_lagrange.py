import sympy as sp

x = sp.Symbol('x')

def lagrange_poly(nodo, nodos):
    L = 1
    for i in range(len(nodos)):
        if i != nodo:
            L *= (x - nodos[i])/(nodos[nodo] - nodos[i])
    return L

def f(n):
    return 1/(1+(25*n)**2)

num_nodos = int(input("Ingrese el número de nodos: "))
nodos = [i*sp.Rational(2, num_nodos-1)-1 for i in range(num_nodos)]

def lagrange_interpolation(funcion, nodos):
    polinomio = 0
    for i in range(len(nodos)):
        polinomio += funcion(nodos[i], nodos)*f(nodos[i])
    return polinomio

f = 1/(1+25*x**2)
polinomio = lagrange_interpolation(lagrange_poly, nodos)
print(polinomio)
