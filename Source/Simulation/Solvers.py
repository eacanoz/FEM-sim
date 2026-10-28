import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Source.Simulation.DirectSolver import DirectSolver
from Source.Simulation.IterativeSolver import IterativeSolver
from Source.enums import SolverType, StudyType, ProblemType
from Source.logger_config import sim_logger


def tictoc(method):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            method(*args, **kwargs)
            t2 = time.time() - t1
            #print(f'method finished on {round(t2, 3)} seconds')
            sim_logger.info(f'method finished on {round(t2, 3)} seconds')
            return t2
        
        return wrapper

def _same_sparse(a, b):

    """True if two sparse matrices have the same structure and values."""

    if b is None or a.shape != b.shape or a.nnz != b.nnz:
        return False

    return (np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
            and np.array_equal(a.data, b.data))


class modelSolver():

    """Class responsible for solving the model based on the physics and mesh. 
    It includes methods for assembling the global system, solving linear and nonlinear problems, and handling transient simulations.
    """

    def __init__(self, model):

        """Initializes the model solver with a reference to the model and sets up necessary attributes."""

        self.model = model

        self.A = None
        self.b = None
        self.M = None

        self.x0 = np.array([])
        self.sol = None

        self.solverOptions = model.solverOptions
        self.nDoF = model.mesh.getNoN()
        self.nVar = None

        self.timeVector = None


    def construcProblem(self):

        """Assembles the global system of equations for the model based on the physics and mesh."""
 
        self.A, self.b = self.model.assembleGlobalSystem(self.solverOptions)

        if self.model.solverOptions['Study'] == StudyType.transient:
            self.M = self.model.assembleMassMatrix(self.solverOptions)
            
            self.M = self.M.tocsc()

        self.A = self.A.tocsc()


    @tictoc
    def solve(self):

        """Solves the model based on the configured solver options and updates the solution in the model."""

        solDict = {}

        self.getFieldVariables()

        if self.model.solverOptions['Study'] == StudyType.steady_state:
            self.steadyStateSolver()
        elif self.model.solverOptions['Study'] == StudyType.transient:
            self.transientSolver()

        for fieldVar in self.fieldVariables:
            solDict[fieldVar] = self.model.physics.var[fieldVar].getFieldValues(self.solverOptions['Study'])

        self.model.sol = solDict


    def steadyStateSolver(self):

        """Solves the steady-state problem for the model based on the configured solver options."""

        if self.model.solverOptions['Type'] == ProblemType.linear:
            self.linearSolver()
        elif self.model.solverOptions['Type'] == ProblemType.nonlinear:
            self.nonlinearSolver()

    def transientSolver(self):

        """Theta-method with a fixed time step: theta = 1 is implicit Euler, theta = 0.5 Crank-Nicolson.

        Each step assembles the system at the current state, so nonlinear BCs are evaluated at the
        start of the step. The step matrix is factorized again only when it changes.
        Dirichlet nodes keep their prescribed values (the mass matrix is zeroed on those rows/columns).
        """

        theta = float(self.solverOptions['Time Discretization'])
        total_time = float(self.solverOptions['Time'])
        dt_requested = float(self.solverOptions['Time Step'])

        n_steps = max(1, int(np.ceil(total_time / dt_requested - 1e-12)))
        dt = total_time / n_steps

        sim_logger.info(f"Theta = {theta}, steps = {n_steps}, dt = {dt:.6g}")

        self.getInitialField()

        dirichlet = self.model.dirichletDofs()

        # Initial state satisfies the Dirichlet conditions
        A, b = self.model.assembleGlobalSystem(self.solverOptions)
        x = self.x0.copy()
        x[dirichlet] = b[dirichlet]
        self.updateSolution(x)

        free = sp.diags((~dirichlet).astype(float))
        M = (free @ self.model.assembleMassMatrix(self.solverOptions).tocsr() @ free).tocsr()

        Y = np.zeros((len(x), n_steps + 1))
        Y[:, 0] = x

        lu, S_prev = None, None

        for step in range(n_steps):

            A, b = self.model.assembleGlobalSystem(self.solverOptions)

            S = (M + theta * dt * A).tocsc()

            if lu is None or not _same_sparse(S, S_prev):
                lu = spla.splu(S)
                S_prev = S

            rhs = M @ x - (1.0 - theta) * dt * (A @ x) + dt * b

            x = lu.solve(rhs)
            self.updateSolution(x)

            Y[:, step + 1] = x

        self.timeVector = np.linspace(0.0, total_time, n_steps + 1)

        layout = self.model.layout
        for name in layout.names:
            self.model.physics.var[name].updateTimeValues(Y[layout.dofs(name)[:, 0], :])

    def linearSolver(self):

        """Solves the linear problem for the model based on the configured solver options.
        It constructs the global system and uses either a direct or iterative solver to find the solution.
        """

        self.construcProblem()
        self.getInitialField()

        if self.model.solverOptions['Method'] == SolverType.direct:
            solutionMethod = DirectSolver(self.A, self.b, self.x0, self.model.solverOptions)

            self.sol = solutionMethod.solve()


        elif self.model.solverOptions['Method'] == SolverType.iterative:
            solutionMethod = IterativeSolver(self.A, self.b, self.x0, self.model.solverOptions)

            self.sol = solutionMethod.solve()

        self.updateSolution(self.sol)


    def nonlinearSolver(self):

        """Solves the nonlinear problem for the model based on the configured solver options.
        It constructs the global system, computes the Jacobian, and uses a Newton solver to find the solution iteratively until convergence."""

        self.getInitialField()

        sol = self.newtonSolver()

        self.updateSolution(sol)



    def _solve_newton_step(self, J, rhs):

        """Solves J dX = rhs with the configured linear solver."""

        if self.model.solverOptions['Method'] == SolverType.direct:
            return DirectSolver(J, rhs, self.x0, self.model.solverOptions).solve()

        return IterativeSolver(J, rhs, self.x0, self.model.solverOptions).solve()

    def newtonSolver(self):

        """Newton-Raphson for the nonlinear problem, with a backtracking line search on ||R||.

        Stops when ||R|| <= rtol * ||R0|| or ||R|| <= atol. The residual and the tangent come from the
        model; trial states are written to the field, and the last accepted state is kept.
        """

        sim_logger.info("------- Starting Newton-Raphson nonlinear solver -------")

        rtol = 1e-8
        atol = 1e-10
        max_iterations = 200
        lam_min = 1.0 / 1024.0

        sim_logger.info(f"Relative tolerance: {rtol:.1e}, Absolute tolerance: {atol:.1e}, "
                        f"Max Number of Iterations: {max_iterations}")

        header = f"| {'Iter':^6} | {'||R||':^15} | {'||R||/||R0||':^15} | {'Alpha (Step)':^12}"
        separator = "-" * len(header)

        sim_logger.info(separator)
        sim_logger.info(header)
        sim_logger.info(separator)

        x = self.x0.copy()

        J, R = self.model.assembleGlobalSystemNonLinear(self.solverOptions)
        norm_R0 = np.linalg.norm(R)
        norm_R = norm_R0

        iteration = 0

        while norm_R > atol and norm_R > rtol * norm_R0 and iteration < max_iterations:

            iteration += 1

            dX = self._solve_newton_step(J, -R)

            lam = 1.0
            while True:

                x_try = x + lam * dX
                self.updateSolution(x_try)

                J_try, R_try = self.model.assembleGlobalSystemNonLinear(self.solverOptions)
                norm_try = np.linalg.norm(R_try)

                if norm_try < (1.0 - 1e-4 * lam) * norm_R or lam <= lam_min:
                    break

                lam *= 0.5

            x, J, R, norm_R = x_try, J_try, R_try, norm_try

            sim_logger.info(f"| {iteration:^6} | {norm_R:^15.4e} | {norm_R / norm_R0:^15.4e} | {lam:^12.4f}")

        sim_logger.info(separator)

        if norm_R <= atol or norm_R <= rtol * norm_R0:
            sim_logger.info(f"Succefully converged in {iteration} iterations.")
        else:
            sim_logger.warning("Solver did not converge at the specified tolerance and number of iterations.")

        self.x0 = x
        self.updateSolution(x)

        return x

    def updateSolution(self, x):

        """Writes the solution vector x into the fields of the model."""

        for name, values in self.model.layout.split(x).items():
            self.model.physics.var[name].updateField(values)

    def getFieldVariables(self):

        """Retrieves the field variables from the model's physics and stores them in an attribute for later use."""

        self.fieldVariables = self.model.physics.var.keys()

        self.nVar = len(self.fieldVariables)

    def getInitialField(self):

        """Collects the current values of every field into the solution vector x0 used by the solvers."""

        self.x0 = self.model._state()
