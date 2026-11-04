import time
from dataclasses import dataclass, field

import numpy as np

from Source.enums import SolverType, DirectSolvers, IterativeSolvers, Preconditioners


@dataclass
class Candidate:

    """A linear-solver configuration: a label and the keywords for Model.solverConfiguration."""

    label: str
    options: dict = field(default_factory=dict)

    @property
    def direct(self):
        return self.options.get('Method', SolverType.direct) == SolverType.direct


@dataclass
class Result:

    label: str
    seconds: float = np.nan
    iterations: int = 0           # Krylov iterations summed over every linear solve
    solves: int = 0               # linear solves (Newton steps, time steps)
    converged: bool = False
    error: float = np.nan         # largest field difference with the reference, relative to the reference size
    note: str = ''


def direct(label, solver):
    return Candidate(label, dict(Method=SolverType.direct, Solver=solver))


def krylov(label, solver, prec, **extra):
    return Candidate(label, dict(Method=SolverType.iterative, Solver=solver, prec=prec, **extra))


# Candidates worth trying for each kind of system
CANDIDATES = {
    # symmetric positive definite (diffusion): CG is valid
    'spd': [direct('PARDISO', DirectSolvers.PARDISO),
            direct('SuperLU', DirectSolvers.SuperLU),
            krylov('CG+AMG', IterativeSolvers.CG, Preconditioners.AMG),
            krylov('CG+iLU', IterativeSolvers.CG, Preconditioners.iLU),
            krylov('CG+Jacobi', IterativeSolvers.CG, Preconditioners.Jacobi),
            krylov('GMRES+AMG', IterativeSolvers.GMRES, Preconditioners.AMG)],
    # convection, reaction, coupled scalars: not symmetric
    'general': [direct('PARDISO', DirectSolvers.PARDISO),
                direct('SuperLU', DirectSolvers.SuperLU),
                krylov('GMRES+iLU', IterativeSolvers.GMRES, Preconditioners.iLU),
                krylov('BiCGSTAB+iLU', IterativeSolvers.BiCGSTAB, Preconditioners.iLU),
                krylov('GMRES+AMG', IterativeSolvers.GMRES, Preconditioners.AMG),
                krylov('BiCGSTAB+Jacobi', IterativeSolvers.BiCGSTAB, Preconditioners.Jacobi)],
    # velocity-pressure saddle point
    'flow': [direct('PARDISO', DirectSolvers.PARDISO),
             direct('SuperLU', DirectSolvers.SuperLU),
             krylov('GMRES+Block', IterativeSolvers.GMRES, Preconditioners.SaddlePoint, tol=1e-8),
             krylov('GMRES+iLU', IterativeSolvers.GMRES, Preconditioners.iLU, tol=1e-8)],
}


def benchmark(build, base, candidates, field_name, repeat=1):

    """Solves the same problem with every candidate and measures it.

    build(): returns a new model (boundary conditions and fields included). base: keywords common to all candidates
    (Study, Type, time stepping...). The reference is the first direct candidate; error is the largest difference of
    `field_name` with it over the largest absolute value of the reference. The time is the best of `repeat` runs and
    includes assembly, which the candidates share; iterations and converged come from the Krylov solves.
    """

    results, reference = [], None

    for candidate in candidates:
        result = Result(candidate.label)
        best = np.inf
        try:
            for _ in range(repeat):
                model = build()
                model.solverConfiguration(**{**base, **candidate.options})
                start = time.perf_counter()
                model.solve()
                best = min(best, time.perf_counter() - start)
            values = np.asarray(model.sol[field_name], dtype=float)
        except Exception as error:
            result.note = f"{type(error).__name__}: {str(error)[:60]}"
            results.append(result)
            continue

        result.seconds = best
        result.iterations = int(sum(model.solver_iterations))
        result.solves = len(model.solver_iterations)
        result.converged = model.solver_unconverged == 0
        if model.nonlinear_converged is not None:
            result.converged &= bool(model.nonlinear_converged)

        if reference is None and candidate.direct:
            reference = values
        if reference is not None:
            result.error = float(np.max(np.abs(values - reference)) / max(np.max(np.abs(reference)), 1e-300))
        results.append(result)

    return results


def best_of(results, max_error=1e-6):

    """Fastest converged result whose error is below max_error."""

    valid = [r for r in results if r.converged and np.isfinite(r.seconds) and (not np.isfinite(r.error) or r.error <= max_error)]
    return min(valid, key=lambda r: r.seconds) if valid else None


def format_table(title, results, max_error=1e-6):

    best = best_of(results, max_error)
    lines = [title, f"   {'solver':<16}{'time [s]':>10}{'Krylov it':>11}{'solves':>8}{'error':>10}  "]
    for r in results:
        if r.note:
            lines.append(f"   {r.label:<16}{'-':>10}{'-':>11}{'-':>8}{'-':>10}  {r.note}")
            continue
        flag = ('  <- fastest valid' if r is best else '') + ('' if r.converged else '  NO CONVERGE')
        lines.append(f"   {r.label:<16}{r.seconds:>10.3f}{r.iterations:>11d}{r.solves:>8d}{r.error:>10.1e}{flag}")
    return "\n".join(lines)
