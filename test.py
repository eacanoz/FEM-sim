import numpy as np
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt

# Definimos la matriz A y el vector b
A = np.array([[10, -1, 2], [-1, 11, -1], [2, -1, 10]])
b = np.array([6, 25, -11])

# Definimos la función de callback para registrar la información
# de cada iteración
iter_count = 0
error_list = []
iter_list = []

def callback(xk):
    global iter_count
    global error_list
    global iter_list
    
    # Calculamos el error actual
    error = np.linalg.norm(np.dot(A, xk) - b)
    
    # Almacenamos el error y el número de iteración
    error_list.append(error)
    iter_list.append(iter_count)
    
    # Incrementamos el número de iteración
    iter_count += 1

# Llamamos a la función bicgstab y utilizamos la función de callback
x, exit_code = bicgstab(A, b, callback=callback)

# Graficamos la convergencia
plt.plot(iter_list, error_list)
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.title('Convergencia de Bicgstab')
plt.show()
