import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Source.Simulation.DirectSolver import FactorizedSolver, _same_sparse
from Source.Simulation.IterativeSolver import IterativeSolver, build_preconditioner, saddle_point_preconditioner
from Source.enums import SolverType, StudyType, ProblemType, Preconditioners
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
            if self.solverOptions.get('Preconditioner') == Preconditioners.SaddlePoint:
                preconditioner = self._saddle_point(A_ff, free)
            else:
                preconditioner = build_preconditioner(A_ff, self.solverOptions.get('Preconditioner'))
            self._iterative_cache = (A_ff, preconditioner)

        x0 = None if guess is None else guess[free]
        solver = IterativeSolver(A_ff, rhs, x0, self.solverOptions, preconditioner=preconditioner)
        x[free] = solver.solve()
        self.iterations.append(solver.iterations)

        return x

    def _saddle_point(self, A_ff, free):

        """Block preconditioner of the flow: velocity and pressure of the free dofs, Schur diagonal from the flow physics, the
        other fields with an incomplete LU."""

        model = self.model
        flow = next((part for part in getattr(model.physics, 'parts', {'flow': model.physics}).values()
                     if getattr(part, 'computes_velocity', False)), None)
        if flow is None:
            raise ValueError("El precondicionador de bloques necesita un flujo: campos u y p.")

        number = np.cumsum(free) - 1
        u, p = model.layout.dofs('u').ravel(), model.layout.dofs('p')[:, 0]
        schur = flow.schur_diagonal(model._system(), model._state())

        # Near-null space of the viscous block: the rigid motions of the velocity
        xy = model.geometry.node_xy
        
        modes = np.zeros((2 * len(xy), 3))
        modes[0::2, 0], modes[1::2, 1] = 1.0, 1.0
        modes[0::2, 2], modes[1::2, 2] = -xy[:, 1], xy[:, 0]

        rest = np.concatenate([model.layout.dofs(name).ravel() for name in model.layout.names if name not in ('u', 'p')] or [[]]).astype(int)

        return saddle_point_preconditioner(A_ff, number[u[free[u]]], number[p[free[p]]], schur[free[p]], modes[free[u]],
                                           number[rest[free[rest]]])

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

        system = self.model._system()
        if scheme != 'theta' or not system.is_linear or system.time_dependent:
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

    def _store_history(self, Y, total_time, n_steps, times=None):

        self.timeVector = np.linspace(0.0, total_time, n_steps + 1) if times is None else np.asarray(times)

        layout = self.model.layout
        for name in layout.names:
            d = layout.dofs(name)
            self.model.physics.var[name].updateTimeValues(Y[d[:, 0], :] if d.shape[1] == 1 else Y[d, :])

    def _nonlinear_transient(self, scheme, theta, n_steps, dt, total_time):

        """Fully implicit time stepping with Newton iterations inside every step.

        Semi-discrete system M x' + R(x) = 0 (R: residual of the System, nonlinear terms included). Each step solves

            theta:  F(x) = M (x - x_n) / dt + theta R(x) + (1 - theta) R(x_n) = 0
            BDF2:   F(x) = M (alpha x + beta) + R(x) = 0,  alpha = (1 + 2 r) / ((1 + r) dt), r = dt / dt_old
                    beta = (-(1 + r) x_n + r^2 / (1 + r) x_{n-1}) / dt                 (first step: implicit Euler)

        with the tangent dF/dx = alpha M + theta J. Algebraic rows (no mass: the incompressibility constraint) are always
        fully implicit, F = R(x), and the algebraic unknowns (the pressure) enter the old-level residual with their new
        values: averaging a constraint or its multiplier between two time levels (Crank-Nicolson) makes the pressure of
        an incompressible flow oscillate and blow up.

        The step is published in system.time = {alpha, beta (x' = alpha x + beta), theta, x_old, dt}, so that the
        residual-based stabilization (PSPG/SUPG of the flow, SUPG of a convected field) uses the time-discrete residual
        of the step, x' + theta R_space(x) + (1 - theta) R_space(x_n), once and at full weight; the old-level residual is
        evaluated with {'old_level': True} and leaves that stabilization out. Dirichlet dofs keep their values. Newton
        stops when the scaled residual drops by 1e-8 or below 1e-10, or when the update moves the differential dofs of
        every field by less than 1e-10 of its size: the round-off of the residual grows with alpha and a good initial guess
        starts near that floor, so the residual alone can never report the step as solved. The algebraic unknowns do not
        count: they follow the others, and a pressure that is exactly zero has no size to compare with.

        With theta < 1 the implicit pressure of the step balances the time-centred momentum, so it belongs to t_n + theta dt,
        an error dt (1 - theta) dp/dt in a value reported at t_{n+1}. The velocity does not notice. The history reports the
        algebraic unknowns at t_{n+1}, p^{n+1} + (1 - theta) (p^{n+1} - p^n), from the second step on; the iteration keeps
        its own values. Crank-Nicolson is thus second order in the pressure as well.

        Modified Newton (option 'Reuse Tangent'): the tangent and its factorization stay in use, also from one time step
        to the next, while each iteration reduces the residual at least by half; otherwise the iteration is discarded
        and the tangent rebuilt. An assembly with its factorization costs several residual evaluations, and the state
        changes little between iterations of a step. A change of alpha by more than 30 % or of theta also rebuilds it. A
        system without nonlinear terms assembles once, however dt moves: a change of alpha by more than 10 % only adds
        alpha M again and factorizes.

        Adaptive steps (BDF2, option 'Adaptive Tolerance'): the local truncation error comes from the difference with a
        quadratic extrapolation of the last three states, h (h + k) / ((2 h + k) (h + k + l)) (x - x_pred) for the steps
        h, k, l, measured per field relative to its size, the largest value it has reached and the amplitude of its time
        functions (a field that starts at zero has no size of its own), over the differential dofs. A step above the
        tolerance is repeated; dt follows (tol / e)^(1/3), but a growth below 20 % is left out, because keeping dt keeps the
        factorization of the tangent. dt never more than doubles (zero stability needs a ratio below 1 + sqrt(2)), nor
        exceeds 'Adaptive Max Step'. The first steps, without enough history, use the given time step: choose it small.
        """

        system = self.model._system()
        dirichlet = system.is_dirichlet
        timed = system.time_dependent
        tolerance = self.solverOptions.get('Adaptive Tolerance')

        if timed:
            system.set_time(0.0)

        x = self.x0.copy()
        x[dirichlet] = system.dirichlet_value[dirichlet]
        self.updateSolution(x)

        free = sp.diags((~dirichlet).astype(float))
        # A moving boundary value has x' != 0: its column of the mass matrix stays
        M = (free @ system.mass().tocsr() @ (sp.identity(len(x)) if timed else free)).tocsr()
        algebraic = (np.diff(M.indptr) == 0) & ~dirichlet
        differential_dofs = ~(dirichlet | algebraic)
        differential = sp.diags(differential_dofs.astype(float))
        to_algebraic = sp.diags(algebraic.astype(float))

        times = [0.0]
        history = [x.copy()]
        x_prev, dt_prev = None, None
        self.step_iterations = []
        self.rejected_steps = 0

        rtol, atol, max_iterations = 1e-8, 1e-10, 30
        weights = system.relaxation_weights(self.solverOptions.get('Relaxation'))
        reuse = self.solverOptions.get('Reuse Tangent', True)

        self.tangent_assemblies = 0
        J, J_core, J_alpha, J_theta = None, None, None, None

        fields = [self.model.layout.dofs(name).ravel() for name in self.model.layout.names]

        amplitude = np.zeros(len(x))
        for dofs, values, _ in system._timed_dirichlet:
            amplitude[dofs] = np.abs(values)
        sizes = [max(np.abs(x[dofs]).max(), amplitude[dofs].max()) for dofs in fields]

        def negligible(update, state):
            return all(np.abs(update[dofs[differential_dofs[dofs]]]).max(initial=0.0)
                       <= 1e-10 * max(np.abs(state[dofs]).max(), 1e-30) for dofs in fields)

        clock = time.time()
        step = 0
        try:
            while (step < n_steps) if tolerance is None else (times[-1] < total_time * (1.0 - 1e-12)):

                t_n = times[-1]
                if tolerance is None:
                    t_new = (step + 1) * dt
                else:
                    dt = min(dt, total_time - t_n)
                    t_new = t_n + dt

                if timed:
                    system.set_time(t_new)

                x_n = x.copy()
                if scheme == 'BDF2' and x_prev is not None:
                    r = dt / dt_prev
                    alpha = (1.0 + 2.0 * r) / ((1.0 + r) * dt)
                    beta = (-(1.0 + r) * x_n + r ** 2 / (1.0 + r) * x_prev) / dt
                    th = 1.0
                    x = x_n + r * (x_n - x_prev)                               # extrapolated initial guess
                else:
                    alpha, beta = 1.0 / dt, -x_n / dt
                    th = 1.0 if scheme == 'BDF2' else theta
                x[dirichlet] = system.dirichlet_value[dirichlet]

                weight = np.where(algebraic, 1.0, th)                          # theta per row
                context = {'alpha': alpha, 'beta': beta, 'theta': th, 'x_old': x_n, 'dt': dt}

                def at_old_level(evaluate, state):
                    # (1 - theta) R at the old level, with the algebraic unknowns (pressure) of the new level:
                    # they act as Lagrange multipliers and stay fully implicit (no oscillating pressure)
                    old = x_n.copy()
                    old[algebraic] = state[algebraic]
                    system.time = {'old_level': True}
                    if timed:
                        system.set_time(t_n)
                    value = evaluate(old)
                    if timed:
                        system.set_time(t_new)
                    system.time = context
                    return value

                def step_residual(state):
                    system.time = context
                    F = M @ (alpha * state + beta) + weight * system.residual(state)
                    if th < 1.0:
                        F = F + (1.0 - th) * (differential @ at_old_level(system.residual, state))
                    return F

                def step_tangent(state):
                    nonlocal J_core
                    if not (system.is_linear and J_theta == th):
                        self.tangent_assemblies += 1
                        system.time = context
                        tangent, _ = system.residual_and_tangent(state)
                        J_core = sp.diags(weight) @ tangent
                        if th < 1.0:
                            J_old = at_old_level(lambda old: system.residual_and_tangent(old)[0], state)
                            J_core = J_core + (1.0 - th) * (differential @ J_old @ to_algebraic)
                    return (alpha * M + J_core).tocsr()

                # The tangent belongs to this state?
                change = np.inf if J is None else abs(alpha / J_alpha - 1.0)
                fresh = J is None or J_theta != th or not reuse or (not system.is_linear and change > 0.3)
                if fresh or change > 0.1 and system.is_linear:
                    J, J_alpha, J_theta = step_tangent(x), alpha, th
                    fresh = True

                F = step_residual(x)
                scale = np.maximum(np.abs(J.diagonal()), 1.0) * system.field_scales(x)
                norm0 = norm = np.linalg.norm(F / scale)
                iteration = 0
                settled = False

                while norm > atol and norm > rtol * norm0 and iteration < max_iterations:
                    iteration += 1
                    dx = self._solve_system(J, -F)

                    if not fresh:
                        # Old tangent: keep the step only if it contracts well
                        x_try = system.clip_positive(x + weights * dx)
                        F_try = step_residual(x_try)
                        norm_try = np.linalg.norm(F_try / scale) if system.is_admissible(x_try) else np.nan
                        settled = np.isfinite(norm_try) and negligible(x_try - x, x)
                        if settled:
                            x, F, norm = x_try, F_try, norm_try
                            break
                        if np.isfinite(norm_try) and norm_try <= 0.5 * norm:
                            x, F, norm = x_try, F_try, norm_try
                            continue
                        J, J_alpha, J_theta, fresh = step_tangent(x), alpha, th, True
                        dx = self._solve_system(J, -F)

                    lam = 1.0
                    while True:
                        x_try = system.clip_positive(x + lam * weights * dx)   # positive fields stay positive
                        F_try = step_residual(x_try) if system.is_admissible(x_try) else np.full(len(x), np.nan)
                        norm_try = np.linalg.norm(F_try / scale)
                        if (np.isfinite(norm_try) and norm_try < (1.0 - 1e-4 * lam) * norm) or lam <= 1.0 / 64:
                            break
                        lam *= 0.5
                    settled = lam == 1.0 and negligible(x_try - x, x)
                    x, F, norm = x_try, F_try, norm_try
                    if settled and np.isfinite(norm):
                        break
                    fresh = not reuse
                    if fresh:
                        J, J_alpha, J_theta = step_tangent(x), alpha, th

                finite = np.isfinite(norm)
                solved = finite and (norm <= atol or norm <= rtol * norm0 or settled)

                if tolerance is not None and not solved and dt > 1e-10 * total_time:
                    self.rejected_steps += 1                                   # repeat the step, shorter
                    x, dt = x_n, 0.5 * dt
                    continue

                if not finite:                                                 # NaN passes the stopping test
                    sim_logger.warning(f"Paso {step + 1}: el residuo no es finito. La simulacion se detiene en "
                                       f"t = {t_n:.6g}; se conservan los pasos anteriores.")
                    x = x_n
                    self.updateSolution(x)
                    break

                if not solved:
                    sim_logger.warning(f"Paso {step + 1}: Newton no convergio en {max_iterations} iteraciones "
                                       f"(residuo relativo {norm / max(norm0, 1e-300):.1e}).")

                dt_next = dt
                if tolerance is not None and len(times) >= 3:
                    h, k, l = dt, t_n - times[-2], times[-2] - times[-3]
                    t0, t1, t2 = t_n, times[-2], times[-3]
                    predicted = (history[-1] * (t_new - t1) * (t_new - t2) / ((t0 - t1) * (t0 - t2))
                                 + history[-2] * (t_new - t0) * (t_new - t2) / ((t1 - t0) * (t1 - t2))
                                 + history[-3] * (t_new - t0) * (t_new - t1) / ((t2 - t0) * (t2 - t1)))
                    error = np.abs(h * (h + k) / ((2 * h + k) * (h + k + l)) * (x - predicted))
                    relative = 0.0
                    for dofs, size in zip(fields, sizes):
                        moving = dofs[differential_dofs[dofs]]
                        if moving.size:
                            relative = max(relative, error[moving].max() / max(size, np.abs(x[dofs]).max(), 1e-30))

                    factor = 0.9 * (tolerance / max(relative, 1e-30)) ** (1.0 / 3.0)
                    if relative > tolerance and dt > 1e-10 * total_time:
                        self.rejected_steps += 1
                        x, dt = x_n, dt * max(factor, 0.2)
                        continue
                    dt_next = dt if factor < 1.2 else dt * min(factor, 2.0)
                    dt_next = min(dt_next, self.solverOptions.get('Adaptive Max Step') or np.inf)

                self.step_iterations.append(iteration)
                sizes = [max(size, np.abs(x[dofs]).max()) for size, dofs in zip(sizes, fields)]
                x_prev, dt_prev = x_n, dt
                self.updateSolution(x)
                times.append(t_new)
                reported = x.copy()
                if th < 1.0 and step > 0:
                    reported[algebraic] = x[algebraic] + (1.0 - th) * (x[algebraic] - x_n[algebraic])
                history.append(reported)
                step += 1
                if tolerance is None:
                    self._report_progress(step - 1, n_steps, dt, clock)
                elif step % 50 == 0:
                    sim_logger.info(f"Paso {step}: t = {t_new:.6g} de {total_time:.6g}, dt = {dt:.3g}, "
                                    f"{time.time() - clock:.1f} s")
                dt = dt_next
        finally:
            system.time = None

        if self.step_iterations:
            sim_logger.info(f"Newton per step: mean {np.mean(self.step_iterations):.1f}, "
                            f"max {np.max(self.step_iterations)} iterations, {self.tangent_assemblies} tangents")

        self._store_history(np.column_stack(history), times[-1], len(times) - 1, times)

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

        if self.solverOptions.get('Globalization', 'newton') == 'pseudo-transient':
            sol = self.pseudoTransientSolver()
        else:
            sol = self.newtonSolver()

        self.updateSolution(sol)



    def _solve_newton_step(self, J, rhs):

        """Solves J dX = rhs with the configured linear solver."""

        return self._solve_system(J, rhs)

    def newtonSolver(self):

        """Newton-Raphson for the nonlinear problem, with a backtracking line search on the scaled residual.

        The residual is scaled row by row with the diagonal of the initial tangent (never below 1), which
        expresses every equation in the units of its own unknown, and then with the size of its field
        (System.field_scales), which makes it relative. Without them, a field with large numbers (a
        temperature of 300, omega ~ 1e6) would dominate the norm over a field of order 1 (a concentration).
        Only the norm is scaled: the Newton step is computed with the unscaled system.

        Stops when ||R|| <= rtol * ||R0|| or ||R|| <= atol (scaled norms), or when a full Newton step moves the unknowns with
        an equation of their own by less than 1e-10 of the size of their field (System.negligible_update): a state that
        starts near the solution is at the round-off floor of the residual already. The residual and the tangent come from
        the model; trial states are written to the field, and the last accepted state is kept.
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

        system = self.model._system()
        J, R = system.hold_clipped(J, R, x)
        scale = np.maximum(np.abs(J.diagonal()), 1.0) * system.field_scales(x)
        weights = system.relaxation_weights(self.solverOptions.get('Relaxation'))

        def scaled_norm(residual):
            return np.linalg.norm(residual / scale)

        norm_R0 = scaled_norm(R)
        norm_R = norm_R0

        iteration = 0
        diverged = False
        settled = False

        while norm_R > atol and norm_R > rtol * norm_R0 and iteration < max_iterations and not settled:

            iteration += 1

            dX = self._solve_newton_step(J, -R)

            lam = 1.0
            while True:

                x_try = system.clip_positive(x + lam * weights * dX)        # positive fields stay positive
                self.updateSolution(x_try)

                J_try, R_try = self.model.assembleGlobalSystemNonLinear(self.solverOptions)
                J_try, R_try = system.hold_clipped(J_try, R_try, x_try)
                norm_try = scaled_norm(R_try) if system.is_admissible(x_try) else np.nan

                finite = np.isfinite(norm_try)

                if finite and (norm_try < (1.0 - 1e-4 * lam) * norm_R or lam <= lam_min):
                    break

                if not finite and lam <= lam_min:
                    diverged = True
                    break

                lam *= 0.5

            if diverged:
                break

            settled = lam == 1.0 and system.negligible_update(x_try - x, x)  # what moved: a clipped dof asks for more, in vain
            x, J, R, norm_R = x_try, J_try, R_try, norm_try

            sim_logger.info(f"| {iteration:^6} | {norm_R:^15.4e} | {norm_R / norm_R0:^15.4e} | {lam:^12.4f}")

        sim_logger.info(separator)

        self.nonlinear_converged = bool(norm_R <= atol or norm_R <= rtol * norm_R0 or settled)

        if self.nonlinear_converged:
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

    def pseudoTransientSolver(self):

        """Pseudo-transient continuation (Kelley & Keyes 1998) with switched evolution relaxation (Mulder & Van Leer 1985).

        Each iteration solves (J + D) dx = -R with the steady residual R, so the converged solution is that of the steady
        problem and the pseudo-time only changes the path. D = |diag J| / CFL is a local pseudo-time step: the diagonal
        already measures convection, diffusion and reaction of each dof, D ~ m (|u| / h + nu / h^2) / CFL, which is the
        usual m / dt_local. Constraint rows (the pressure) and Dirichlet rows get no term.

        The CFL follows the residual, CFL *= ||R_old|| / ||R_new||, up to cfl_growth per iteration. It keeps growing by at
        least 5 % through rises below 10 %, which are part of the path: while turbulence develops the residual has to rise,
        and lowering the CFL for it freezes the march (k-omega SST on a flat plate). Larger rises lower it, at most by half.
        When it is large the method is Newton, with quadratic convergence. A step that makes the residual non-finite or more
        than `max_rise` times larger is discarded and retried with a quarter of the CFL. Norms and stopping tests are those
        of newtonSolver.
        """

        sim_logger.info("------- Starting pseudo-transient nonlinear solver -------")

        rtol, atol, max_iterations = 1e-10, 1e-10, self.solverOptions.get('Nonlinear Max Iterations') or 500
        cfl = float(self.solverOptions.get('CFL0', 10.0))
        cfl_min, cfl_max, cfl_growth, max_rise = 1e-6, 1e14, 4.0, 4.0

        system = self.model._system()
        x = self.x0.copy()

        # Boundary values enter at once: Dirichlet rows have no pseudo-time, so otherwise the first step imposes them
        # whatever the CFL (a hot wall in a cold melt changes the viscosity by orders of magnitude and no CFL helps)
        x[system.is_dirichlet] = system.dirichlet_value[system.is_dirichlet]
        self.updateSolution(x)

        J, R = self.model.assembleGlobalSystemNonLinear(self.solverOptions)
        J, R = system.hold_clipped(J, R, x)

        scale = np.maximum(np.abs(J.diagonal()), 1.0) * system.field_scales(x)
        weights = system.relaxation_weights(self.solverOptions.get('Relaxation'))
        evolves = (~(system.is_dirichlet | system.constraint)).astype(float)

        norm_R0 = norm_R = np.linalg.norm(R / scale)
        iteration = 0
        settled = False

        while norm_R > atol and norm_R > rtol * norm_R0 and iteration < max_iterations and cfl >= cfl_min and not settled:

            iteration += 1

            D = evolves * np.maximum(np.abs(J.diagonal()), 1e-12) / cfl
            dX = self._solve_newton_step((J + sp.diags(D)).tocsr(), -R)

            x_try = system.clip_positive(x + weights * dX)                  # positive fields stay positive
            self.updateSolution(x_try)

            J_try, R_try = self.model.assembleGlobalSystemNonLinear(self.solverOptions)
            J_try, R_try = system.hold_clipped(J_try, R_try, x_try)
            norm_try = np.linalg.norm(R_try / scale) if system.is_admissible(x_try) else np.nan

            if not np.isfinite(norm_try) or norm_try > max_rise * norm_R:
                cfl *= 0.25
                sim_logger.info(f"| {iteration:^6} | step rejected ({norm_try:.2e}), CFL -> {cfl:.3e}")
                continue

            ratio = norm_R / max(norm_try, 1e-300)
            if ratio >= 0.9:                                                # small rises are part of the path: go on
                cfl = min(cfl * np.clip(ratio, 1.05, cfl_growth), cfl_max)
            else:
                cfl = cfl * max(ratio, 0.5)
            settled = system.negligible_update(x_try - x, x)                # what moved: a clipped dof asks for more, in vain
            x, J, R, norm_R = x_try, J_try, R_try, norm_try
            # The fields grow along the march (k starts at its freestream value): the norm follows their size
            scale = np.maximum(np.abs(J.diagonal()), 1.0) * system.field_scales(x)
            norm_R = np.linalg.norm(R / scale)

            sim_logger.info(f"| {iteration:^6} | ||R|| {norm_R:.4e} | ||R||/||R0|| {norm_R / norm_R0:.4e} | CFL {cfl:.3e}")

        self.nonlinear_converged = bool(norm_R <= atol or norm_R <= rtol * norm_R0 or settled)

        if self.nonlinear_converged:
            sim_logger.info(f"Pseudo-transient solver converged in {iteration} iterations.")
        else:
            sim_logger.warning(f"Pseudo-transient solver did not converge in {iteration} iterations "
                               f"(relative residual {norm_R / max(norm_R0, 1e-300):.1e}, CFL {cfl:.1e}).")

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
