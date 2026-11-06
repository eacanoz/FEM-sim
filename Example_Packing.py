## Ejemplo: empaque de una cavidad con un fundido compresible (ley pvT de Tait, ff.setDensityLaw)
#
# Placa de 2 mm de espesor y 50 mm de largo, ya llena de fundido a 230 C y presion atmosferica (medio espesor, por simetria). En la
# entrada se aplica la presion de empaque de 50 MPa: el fundido se comprime y entra mas material. Incompresible no entraria nada.
#
#   - pvT: Tait de dos dominios ajustada (Fitting.fit_tait) a las medidas de un PP con 30 % de fibra de vidrio (0.1 a 200 MPa, 20 a 280 C):
#     error rms 0.036 %, transicion T_t = 180.7 C + 0.17 K/MPa
#   - viscosidad: Cross-WLF ajustada (Fitting.fit_cross_wlf) a la curva de viscosidad del mismo material (210 a 250 C, 1 a 7e4 1/s):
#     error rms 0.12 %, n = 0.262, tau* = 17.8 kPa; evaluada a 230 C (isotermo)
#
# Se imprime el avance (presion en el extremo cerrado y masa que ha entrado). Con datos propios: Fitting.fit_cross_wlf(gamma_dot, T, eta)
# para la viscosidad y Fitting.fit_tait(p [Pa], T [C], v [m3/kg]) para la pvT.

import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Rheology import TaitPVT, CrossWLF, Cross
from Source.enums import ElementType, StudyType, ProblemType
from Source.logger_config import sim_logger

class Progress(logging.Handler):

    """El avance del paso adaptativo (cada 50 pasos) y los avisos del resolvedor, con la hora: un estancamiento se ve enseguida."""

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('Paso') or record.levelno >= logging.WARNING:
            print(f"      [{time.strftime('%H:%M:%S')}] {message[:110]}", flush=True)


for handler in list(sim_logger.handlers):
    sim_logger.removeHandler(handler)
sim_logger.addHandler(Progress())
sim_logger.setLevel(logging.INFO)

TAIT = TaitPVT(b1m=1.006e-3, b2m=4.936e-7, b3m=7.989e7, b4m=3.53e-3, b1s=9.073e-4, b2s=1.49e-7, b3s=2.601e8, b4s=2.353e-3,
               b5=453.9, b6=1.709e-7, b7=1.009e-4, b8=4.788e-2, b9=1.553e-8, width=1.0)
T_MELT, P_PACK, T_RAMP = 230.0, 50e6, 5e-3
L, H = 50e-3, 1e-3                                   # largo y medio espesor [m]
VISCOSITY = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6)
mu0 = float(VISCOSITY.zero_shear(T_MELT))
RHO0 = float(TAIT.density(1e5, T_MELT))
print(f"Fundido a {T_MELT} C: rho {RHO0:.1f} kg/m3 a 0.1 MPa, {float(TAIT.density(1e5 + P_PACK, T_MELT)):.1f} a 50 MPa "
      f"(+{100 * (float(TAIT.density(1e5 + P_PACK, T_MELT)) / RHO0 - 1):.2f} %); mu0 {mu0:.0f} Pa s")

mesh = Mesh()
mesh.generate2DMesh((L, H), (60, 4), ElementType.quadrilateral, order=2)
xy = np.array([n.coor for n in mesh.NL])
gate = mesh.boundary_edges(lambda x, y: x < 1e-12)


def model_for(compressible, dt, total, tol=1e-3):
    model = Model(name='empaque', dim=2, mesh=mesh, mat=material('PP-GF', miu=mu0, rho=RHO0), psc=ff)
    flow = model.physics
    # Cross-WLF a T_MELT, con una tasa minima: el fundido queda en reposo y la derivada de Cross en reposo no tiene cota
    flow.setViscosityLaw(Cross(mu0, VISCOSITY.tau_star, VISCOSITY.n, gamma_min=1e-2))
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y > H - 1e-12 or x > L - 1e-12))     # pared y extremo cerrado
    flow.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y < 1e-12))
    flow.addBC_OutletEdges(gate, P_PACK)                                                         # presion de empaque en la entrada
    # en rampa de T_RAMP, como la sube la maquina: un salto en t = 0 obliga al paso adaptativo a bajar sin fin
    flow.setBCTime('u', gate, lambda t: 0.5 * (1.0 - np.cos(np.pi * min(t / T_RAMP, 1.0))))
    if compressible:
        flow.setDensityLaw(TAIT, T=T_MELT, ambient=1e5)
    # paso adaptativo: con un fundido que adelgaza tanto la presion entra mucho mas rapido al principio que lo que dice mu0
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=dt, totalTime=total, timeScheme='BDF2',
                              adaptiveTol=tol)
    return model, flow


# tiempo de la difusion de la presion, con la compresibilidad del fundido y mu0
kappa = float((jnp.log(TAIT.density(1e5 + 1e6, T_MELT)) - jnp.log(TAIT.density(1e5, T_MELT))) / 1e6)
t_diff = 12 * mu0 * kappa * L ** 2 / (2 * H) ** 2
print(f"Compresibilidad {kappa:.2e} 1/Pa: la presion se reparte en ~L^2 / D = {t_diff * 1e3:.1f} ms (mu0; con adelgazamiento, menos)")

total, dt = T_RAMP + 2 * t_diff, 1e-5
start = time.perf_counter()
model, flow = model_for(True, dt, total)
model.solve()
print(f"Compresible: {len(model.timeVector) - 1} pasos (adaptativos) en {time.perf_counter() - start:.0f} s")

t = np.asarray(model.timeVector)
P = np.asarray(model.sol['p'])
U = np.asarray(model.sol['u'])
far = np.argmin(np.abs(xy[:, 0] - L) + np.abs(xy[:, 1]))
disc = model.geometry
mass = np.array([np.sum(disc.detJw * np.asarray(TAIT.density(disc.interpolate(P[:, k]) + 1e5, T_MELT))) for k in range(len(t))])
for k in range(0, len(t), max(1, len(t) // 8)):
    print(f"   t = {t[k] * 1e3:6.2f} ms: p en el extremo cerrado {P[far, k] / 1e6:6.2f} MPa, masa {100 * (mass[k] / mass[0] - 1):5.2f} % mas")

centre = np.flatnonzero(np.abs(xy[:, 1]) < 1e-12)
centre = centre[np.argsort(xy[centre, 0])]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for k, c in zip(np.linspace(0, len(t) - 1, 6).astype(int), plt.cm.viridis(np.linspace(0, 0.9, 6))):
    axes[0].plot(xy[centre, 0] * 1e3, P[centre, k] / 1e6, color=c, label=f't = {t[k] * 1e3:.1f} ms')
axes[0].set_xlabel('x [mm]'); axes[0].set_ylabel('p [MPa]'); axes[0].legend(); axes[0].grid(True)
axes[0].set_title('La presion de empaque entra en la cavidad')
axes[1].plot(t * 1e3, 100 * (mass / mass[0] - 1), '-', label='compresible (Tait)')
axes[1].axhline(0.0, color='k', ls='--', lw=0.8, label='incompresible: no entra masa')
axes[1].set_xlabel('t [ms]'); axes[1].set_ylabel('masa añadida [%]'); axes[1].legend(); axes[1].grid(True)
axes[1].set_title('Masa en la cavidad')
plt.show()
