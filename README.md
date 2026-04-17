# FEM-sim: Finite Element Method Simulator

FEM-sim es un simulador modular desarrollado en Python diseñado para resolver problemas de ingeniería y física mediante el Método de los Elementos Finitos (FEM). El proyecto está estructurado para ser escalable, permitiendo la resolución de problemas lineales y no lineales, tanto en estado estacionario como transitorio.

## 🚀 Características Principales

*   **Soporte Multi-Dimensional:** Generación y gestión de mallas en 1D y 2D (incluyendo elementos cuadrangulares `D2QU4N`).
*   **Solucionadores Avanzados:**
    *   **Lineales:** Métodos directos e iterativos (`BicgStab`, `GMRES`) con precondicionamiento `iLU`.
    *   **No Lineales:** Implementaciones de Newton-Raphson y punto fijo con control de relajación.
    *   **Transitorios:** Integración temporal mediante solvers de ODE (BDF) para problemas dependientes del tiempo.
*   **Motor Matemático Híbrido:** 
    *   Uso de **SymPy** para la generación simbólica de funciones de forma (Lagrange lineal y cuadrática).
    *   Optimización con **Numba** (`@jit`) para acelerar el cálculo numérico y la evaluación de interpolaciones.
*   **Arquitectura Modular:** Separación clara entre el pre-procesamiento (malla y bases), la física del problema y los algoritmos de resolución.

## 📂 Estructura del Proyecto

*   `Source/Pre_processing/`: Gestión de mallas (`Mesh.py`) y funciones de base simbólicas/numéricas (`BasisFunctions.py`).
*   `Source/Simulation/`: Núcleo de los algoritmos de resolución (`Solvers.py`, `IterativeSolver.py`, `DirectSolver.py`).
*   `Test/`: Scripts de validación y pruebas de interpolación.

## 🛠️ Requisitos

El proyecto depende de las siguientes librerías de Python:
*   `numpy` & `scipy`: Operaciones matriciales y solvers algebraicos.
*   `sympy`: Cálculo de funciones de base y derivadas.
*   `numba`: Aceleración de bucles críticos.
*   `matplotlib`: Visualización de mallas y convergencia.
*   `numdifftools`: Cálculo de matrices Jacobianas.

## 🏁 Uso Básico

Para generar una malla 2D y visualizarla, puedes apoyarte en el script de ejemplo `2DFEM.py`:

```python
from Source.Pre_processing.Mesh import Mesh

# Crear malla y definir dimensiones
mesh = Mesh()
mesh.generate2DMesh(dim=(10, 5), div=(20, 10), element_type="D2QU4N")
# ... configurar física y resolver ...
```

## ⚖️ Licencia

Este proyecto está bajo la licencia **GNU General Public License v3.0** (GPLv3). Consulta el archivo LICENSE.md para más detalles.

---
*Desarrollado como una herramienta de simulación científica eficiente y fácil de extender.*