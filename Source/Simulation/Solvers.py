import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Source.Simulation.DirectSolver import FactorizedSolver, _same_sparse
from Source.Simulation.IterativeSolver import IterativeSolver, build_preconditioner
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

        self._iterative_cache = (None, None)   # (matrix, preconditioner) of the last iterative solve
        self.iterations = []                   # Krylov iterations of every iterative solve


    def _factorizer(self):

        """Direct solver kept by the model, so its factorization survives from one solve() to the next."""

        kind = self.solverOptions['Solver']
        factorizer = getattr(self.model, '_factorizer', None)

        if factorizer is None or factorizer.kind != kind:
            factorizer = FactorizedSolver(kind)
            self.model._factorizer = factorizer

        return factorizer

    def _solve_system(self, A, b, guess=None):

        """Solves A x = b with the configured method.

        Direct: reuses the factorization when A has not changed. Iterative: the Dirichlet dofs (identity
        rows) are eliminated first, which leaves the free-dof matrix symmetric for diffusion problems
        (needed by CG and by AMG); the preconditioner is rebuilt only when A changes.
        """

        if self.solverOptions['Method'] == SolverType.direct:
            return self._factorizer().solve(A, b)

        fixed = self.model.dirichletDofs()
        free = ~fixed

        x = np.zeros(len(b))
        x[fixed] = b[fixed] / A.diagonal()[fixed]

        A = A.tocsr()
        A_ff = A[free][:, free].tocsr()
        rhs = b[free] - A[free][:, fixed] @ x[fixed]

        cached_A, preconditioner = self._iterative_cache
        if preconditioner is None or not _same_sparse(A_ff, cached_A):
            preconditioner = build_preconditioner(A_ff, self.solverOptions.get('Preconditioner'))
            self._iterative_cache = (A_ff, preconditioner)

        x0 = None if guess is None else guess[free]
        solver = IterativeSolver(A_ff, rhs, x0, self.solverOptions, preconditioner=preconditioner)
        x[free] = solver.solve()
        self.iterations.append(solver.iterations)

        return x

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
        start of the step. A system without nonlinear terms is assembled and factorized once; otherwise
        the step matrix is factorized again only when it changes.
        Dirichlet nodes keep their prescribed values (the mass matrix is zeroed on those rows/columns).
        """

        theta = float(self.solverOptions['Time Discretization'])
        scheme = self.solverOptions.get('Time Scheme', 'theta')
        total_time = float(self.solverOptions['Time'])
        dt_requested = float(self.solverOptions['Time Step'])

        n_steps = max(1, int(np.ceil(total_time / dt_requested - 1e-12)))
        dt = total_time / n_steps

        sim_logger.info(f"Theta = {theta}, scheme = {scheme}, steps = {n_steps}, dt = {dt:.6g}")

        self.getInitialField()

        if scheme != 'theta' or not self.model._system().is_linear:
            return self._nonlinear_transient(scheme, theta, n_steps, dt, total_time)

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

        linear = self.model._system().is_linear
        S = (M + theta * dt * A).tocsc() if linear else None

        clock = time.time()
        for step in range(n_steps):

            if not linear:
                A, b = self.model.assembleGlobalSystem(self.solverOptions)
                S = (M + theta * dt * A).tocsc()

            rhs = M @ x - (1.0 - theta) * dt * (A @ x) + dt * b

            x = self._solve_system(S, rhs, guess=x)
            self.updateSolution(x)

            Y[:, step + 1] = x
            self._report_progress(step, n_steps, dt, clock)

        self._store_history(Y, total_time, n_steps)

    def _report_progress(self, step, n_steps, dt, clock):

        """Logs the progress of a time loop about every 10 % of the steps: time reached, elapsed and remaining."""

        every = max(1, n_steps // 10)
        done = step + 1
        if done % every and done != n_steps:
            return

        elapsed = time.time() - clock
        remaining = elapsed / done * (n_steps - done)
        sim_logger.info(f"Paso {done}/{n_steps} (t = {done * dt:.6g}, {100 * done / n_steps:.0f} %): "
                        f"{elapsed:.1f} s transcurridos, ~{remaining:.1f} s restantes")

    def _store_history(self, Y, total_time, n_steps):

        self.timeVector = np.linspace(0.0, total_time, n_steps + 1)

        layout = self.model.layout
        for name in layout.names:
            d = layout.dofs(name)
            self.model.physics.var[name].updateTimeValues(Y[d[:, 0], :] if d.shape[1] == 1 else Y[d, :])

    def _nonlinear_transient(self, scheme, theta, n_steps, dt, total_time):

        """Fully implicit time stepping with Newton iterations inside every step.

        Semi-discrete system M x' + R(x) = 0 (R: residual of the System, nonlinear terms included). Each step solves

            theta:  F(x) = M (x - x_n) / dt + theta R(x) + (1 - theta) R(x_n) = 0
            BDF2:   F(x) = M (3 x - 4 x_n + x_{n-1}) / (2 dt) + R(x) = 0          (first step: implicit Euler)

        with the tangent dF/dx = alpha M + theta J. Algebraic rows (no mass: the incompressibility constraint) are always
        fully implicit, F = R(x), and the algebraic unknowns (the pressure) enter the old-level residual with their new
        values: averaging a constraint or its multiplier between two time levels (Crank-Nicolson) makes the pressure of
        an incompressible flow oscillate and blow up.

        The step is published in system.time = {alpha, beta (x' = alpha x + beta), theta, x_old, dt}, so that the
        residual-based stabilization (PSPG/SUPG of the flow, SUPG of a convected field) uses the time-discrete residual
        of the step, x' + theta R_space(x) + (1 - theta) R_space(x_n), once and at full weight; the old-level residual is
        evaluated with {'old_level': True} and leaves that stabilization out. Dirichlet dofs keep their values. Newton
        stops when the scaled residual drops by 1e-8 or below 1e-10.
        """

        system = self.model._system()
        dirichlet = system.is_dirichlet

        x = self.x0.copy()
        x[dirichlet] = system.dirichlet_value[dirichlet]
        self.updateSolution(x)

        free = sp.diags((~dirichlet).astype(float))
        M = (free @ system.mass().tocsr() @ free).tocsr()
        algebraic = (np.diff(M.indptr) == 0) & ~dirichlet

        Y = np.zeros((len(x), n_steps + 1))
        Y[:, 0] = x
        x_prev = None
        self.step_iterations = []

        rtol, atol, max_iterations = 1e-8, 1e-10, 30

        clock = time.time()
        try:
            for step in range(n_steps):

                x_n = x.copy()
                if scheme == 'BDF2' and x_prev is not None:
                    alpha, beta, th = 1.5 / dt, (-2.0 * x_n + 0.5 * x_prev) / dt, 1.0
                    x = 2.0 * x_n - x_prev                                     # extrapolated initial guess
                    x[dirichlet] = x_n[dirichlet]
                else:
                    alpha, beta = 1.0 / dt, -x_n / dt
                    th = 1.0 if scheme == 'BDF2' else theta

                weight = np.where(algebraic, 1.0, th)                          # theta per row
                context = {'alpha': alpha, 'beta': beta, 'theta': th, 'x_old': x_n, 'dt': dt}
                differential = sp.diags((~(dirichlet | algebraic)).astype(float))
                to_algebraic = sp.diags(algebraic.astype(float))

                def step_residual(state):
                    system.time = context
                    J, R = system.residual_and_tangent(state)
                    F = M @ (alpha * state + beta) + weight * R
                    J = alpha * M + sp.diags(weight) @ J
                    if th < 1.0:
                        # (1 - theta) R at the old level, with the algebraic unknowns (pressure) of the new level:
                        # they act as Lagrange multipliers and stay fully implicit (no oscillating pressure)
                        old = x_n.copy()
                        old[algebraic] = state[algebraic]
                        system.time = {'old_level': True}
                        J_old, R_old = system.residual_and_tangent(old)
                        system.time = context
                        F = F + (1.0 - th) * (differential @ R_old)
                        J = J + (1.0 - th) * (differential @ J_old @ to_algebraic)
                    return J.tocsr(), F

                J, F = step_residual(x)
                scale = np.maximum(np.abs(J.diagonal()), 1.0)
                norm0 = norm = np.linalg.norm(F / scale)
                iteration = 0

                while norm > atol and norm > rtol * norm0 and iteration < max_iterations:
                    iteration += 1
                    dx = self._solve_system(J, -F)
                    lam = 1.0
                    while True:
                        J_try, F_try = step_residual(x + lam * dx)
                        norm_try = np.linalg.norm(F_try / scale)
                        if (np.isfinite(norm_try) and norm_try < (1.0 - 1e-4 * lam) * norm) or lam <= 1.0 / 64:
                            break
                        lam *= 0.5
                    x, J, F, norm = x + lam * dx, J_try, F_try, norm_try

                if norm > atol and norm > rtol * norm0:
                    sim_logger.warning(f"Paso {step + 1}: Newton no convergio en {max_iterations} iteraciones "
                                       f"(residuo relativo {norm / max(norm0, 1e-300):.1e}).")

                self.step_iterations.append(iteration)
                x_prev = x_n
                self.updateSolution(x)
                Y[:, step + 1] = x
                self._report_progress(step, n_steps, dt, clock)
        finally:
            system.time = None

        sim_logger.info(f"Newton per step: mean {np.mean(self.step_iterations):.1f}, "
                        f"max {np.max(self.step_iterations)} iterations")

        self._store_history(Y, total_time, n_steps)

    def linearSolver(self):

        """Solves the linear problem for the model based on the configured solver options.
        It constructs the global system and uses either a direct or iterative solver to find the solution.
        """

        self.construcProblem()
        self.getInitialField()

        self.sol = self._solve_system(self.A, self.b, guess=self.x0)

        self.updateSolution(self.sol)


    def nonlinearSolver(self):

        """Solves the nonlinear problem for the model based on the configured solver options.
        It constructs the global system, computes the Jacobian, and uses a Newton solver to find the solution iteratively until convergence."""

        self.getInitialField()

        sol = self.newtonSolver()

        self.updateSolution(sol)



    def _solve_newton_step(self, J, rhs):

        """Solves J dX = rhs with the configured linear solver."""

        return self._solve_system(J, rhs)

    def newtonSolver(self):

        """Newton-Raphson for the nonlinear problem, with a backtracking line search on the scaled residual.

        The residual is scaled row by row with the diagonal of the initial tangent (never below 1), which
        expresses every equation in the units of its own unknown. Without it, a field with large
        numbers (a temperature of 300) would dominate the norm over a field of order 1 (a concentration).
        Only the norm is scaled: the Newton step is computed with the unscaled system.

        Stops when ||R|| <= rtol * ||R0|| or ||R|| <= atol (scaled norms). The residual and the tangent come
        from the model; trial states are written to the field, and the last accepted state is kept.
        """

        sim_logger.info("------- Starting Newton-Raphson nonlinear solver -------")

        rtol = 1e-10
        atol = 1e-10
        max_iterations = 200
        lam_min = 1.0 / 1024.0

        sim_logger.info(f"Relative tolerance: {rtol:.1e}, Absolute tolerance: {atol:.1e}, "
                        f"Max Number of Iterations: {max_iterations}")

        header = f"| {'Iter':^6} | {'||R|| scaled':^15} | {'||R||/||R0||':^15} | {'Alpha (Step)':^12}"
        separator = "-" * len(header)

        sim_logger.info(separator)
        sim_logger.info(header)
        sim_logger.info(separator)

        x = self.x0.copy()

        J, R = self.model.assembleGlobalSystemNonLinear(self.solverOptions)

        scale = np.maximum(np.abs(J.diagonal()), 1.0)

        def scaled_norm(residual):
            return np.linalg.norm(residual / scale)

        norm_R0 = scaled_norm(R)
        norm_R = norm_R0

        iteration = 0
        diverged = False

        while norm_R > atol and norm_R > rtol * norm_R0 and iteration < max_iterations:

            iteration += 1

            dX = self._solve_newton_step(J, -R)

            lam = 1.0
            while True:

                x_try = x + lam * dX
                self.updateSolution(x_try)

                J_try, R_try = self.model.assembleGlobalSystemNonLinear(self.solverOptions)
                norm_try = scaled_norm(R_try)

                finite = np.isfinite(norm_try)

                if finite and (norm_try < (1.0 - 1e-4 * lam) * norm_R or lam <= lam_min):
                    break

                if not finite and lam <= lam_min:
                    diverged = True
                    break

                lam *= 0.5

            if diverged:
                break

            x, J, R, norm_R = x_try, J_try, R_try, norm_try

            sim_logger.info(f"| {iteration:^6} | {norm_R:^15.4e} | {norm_R / norm_R0:^15.4e} | {lam:^12.4f}")

        sim_logger.info(separator)

        if norm_R <= atol or norm_R <= rtol * norm_R0:
            sim_logger.info(f"Succefully converged in {iteration} iterations.")
        elif diverged:
            sim_logger.warning("Newton stopped: the residual is not finite even with the smallest step. "
                               "The last finite state is kept.")
        else:
            sim_logger.warning("Solver did not converge at the specified tolerance and number of iterations.")

        self.newton_iterations = iteration
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
