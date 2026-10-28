"""Runs every automated test suite in Test/ and returns a non-zero exit code if any fails.

Usage (from the project root):
    python run_tests.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

SUITES = [
    "Test/test_analytic.py",
    "Test/test_vectorized.py",
    "Test/test_solvers.py",
    "Test/test_2d_poisson.py",
    "Test/test_foundation.py",
    "Test/test_species.py",
    "Test/test_nonlinear_transient.py",
    "Test/test_nonlinear_diffusion.py",
    "Test/test_coupling.py",
    "Test/test_multifield_diffusion.py",
    "Test/test_convection_supg.py",
    "Test/test_triangles_gmsh.py",
    "Test/test_solver_scaling_output.py",
]


def main():
    env = dict(os.environ, MPLBACKEND="Agg")
    failed = []

    for suite in SUITES:
        print(f"== {suite}")
        result = subprocess.run([sys.executable, os.path.join(ROOT, suite)], cwd=ROOT, env=env,
                                capture_output=True, text=True)

        # Show the test lines only (the simulator logs INFO lines to stderr)
        for line in result.stdout.splitlines():
            if line.startswith(("PASS", "FAIL")) or "passed" in line or line.startswith("   "):
                print(line)

        if result.returncode != 0:
            failed.append(suite)
            print(result.stderr[-2000:])

    print()
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1

    print(f"All {len(SUITES)} suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
