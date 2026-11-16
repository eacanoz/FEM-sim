"""HTML report of a case: one self-contained file (figures embedded as PNG, no external resources) with the parameters, the results,
the fields and the convergence of a solved model.

    from femsim.Post_processing.Report import Report

    report = Report('Dado de rendija', subtitle='POLYFORT, 230 °C')
    report.add_table('Parametros', {'caudal': 1e-4, 'T entrada': 230.0})
    report.add_model(model)                       # summary, field statistics, contours, convergence, probes
    report.add_figure('Mi grafico', fig)          # any matplotlib figure
    report.save('informe.html')

The cases of Source/Cases.py write one with `output.report = true`.
"""

import base64
import html
import io
import json
import os
import tempfile
import time

import numpy as np

STYLE = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --line: #d0d7de; --head: #f6f8fa; --accent: #0969da; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d; --head: #161b22; --accent: #58a6ff; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; } h2 { font-size: 1.15rem; margin: 32px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.sub, .caption { color: var(--muted); font-size: 0.9rem; } .caption { margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 4px 10px; text-align: left; white-space: nowrap; } th { background: var(--head); }
td.num { text-align: right; font-variant-numeric: tabular-nums; } img { max-width: 100%; height: auto; }
pre { background: var(--head); border: 1px solid var(--line); padding: 10px; overflow-x: auto; font-size: 0.85rem; }
"""


def _format(value):

    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)), False
    if isinstance(value, (int, np.integer)):
        return str(int(value)), True
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}", True
    if isinstance(value, (list, tuple, np.ndarray)) and np.size(value) <= 8 and np.ndim(value) <= 1:
        return ', '.join(_format(v)[0] for v in np.asarray(value).tolist()), False
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        return f"({np.size(value) if not isinstance(value, dict) else len(value)} valores)", False
    return str(value), False


class Report:

    def __init__(self, title, subtitle=None):
        self.title, self.subtitle = title, subtitle
        self._parts = []
        self._tail = []                 # sections that go last (the case as written)

    # ----- content -----

    def add_text(self, heading, text):
        self._parts.append(f"<h2>{html.escape(heading)}</h2>\n<p>{html.escape(text)}</p>")

    def add_code(self, heading, text, last=False):
        (self._tail if last else self._parts).append(f"<h2>{html.escape(heading)}</h2>\n<pre>{html.escape(text)}</pre>")

    def add_table(self, heading, data, caption=None):

        """data: a dict (two columns, name and value) or a list of dicts with the same keys (one row each). Numbers are written with 6
        significant figures; long lists are summarized."""

        if isinstance(data, dict):
            rows = ''.join(f"<tr><th>{html.escape(str(k))}</th>{self._cell(v)}</tr>" for k, v in data.items())
            body = f"<table>{rows}</table>"
        else:
            data = list(data)
            keys = list(data[0]) if data else []
            head = ''.join(f"<th>{html.escape(str(k))}</th>" for k in keys)
            rows = ''.join('<tr>' + ''.join(self._cell(row.get(k, '')) for k in keys) + '</tr>' for row in data)
            body = f"<table><tr>{head}</tr>{rows}</table>"

        note = f'<div class="caption">{html.escape(caption)}</div>' if caption else ''
        self._parts.append(f"<h2>{html.escape(heading)}</h2>\n{body}{note}")

    @staticmethod
    def _cell(value):
        text, numeric = _format(value)
        return f'<td class="{"num" if numeric else ""}">{html.escape(text)}</td>'

    def add_figure(self, heading, figure, caption=None, dpi=110):

        """A matplotlib figure, embedded as PNG (the figure is closed)."""

        import matplotlib.pyplot as plt

        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(figure)
        self.add_image(heading, buffer.getvalue(), caption)

    def add_image(self, heading, png_bytes, caption=None):

        data = base64.b64encode(png_bytes).decode('ascii')
        note = f'<div class="caption">{html.escape(caption)}</div>' if caption else ''
        self._parts.append(f'<h2>{html.escape(heading)}</h2>\n<img alt="{html.escape(heading)}" src="data:image/png;base64,{data}">{note}')

    def add_model(self, model, fields=None, max_fields=6, step=None):

        """Summary, statistics and views of a solved model: its fields (default the first max_fields), the convergence table and the probes."""

        import matplotlib
        matplotlib.use('Agg', force=False)
        import matplotlib.pyplot as plt

        if model.sol is None:
            raise RuntimeError("El modelo no tiene solucion: llame a solve() antes del informe.")

        options = model.solverOptions or {}
        summary = {
            'modelo': model.name, 'dimension': f"{model._PD}D" + (' axisimetrico' if model.axisymmetric else ''),
            'nodos': model.mesh.getNoN(), 'elementos': model.mesh.getNoE(), 'fisica': model.physics_status,
            'estudio': getattr(options.get('Study'), 'value', None), 'tipo': getattr(options.get('Type'), 'value', None),
            'metodo': getattr(options.get('Method'), 'value', None),
        }
        if getattr(model, 'newton_iterations', None) is not None:
            summary['iteraciones de Newton'] = model.newton_iterations
            summary['convergio'] = model.nonlinear_converged
        if model.timeVector is not None:
            summary['pasos'] = len(model.timeVector) - 1
            summary['tiempo final'] = float(model.timeVector[-1])
        self.add_table('Modelo', {k: v for k, v in summary.items() if v is not None})

        names = list(fields) if fields else list(model.sol)[:max_fields]
        values = {}
        for name in names:
            array = np.asarray(model.sol[name], dtype=float)
            if model.timeVector is not None and array.shape[-1] == len(model.timeVector):
                array = array[..., -1 if step is None else step]
            values[name] = np.linalg.norm(array, axis=1) if array.ndim == 2 else array

        self.add_table('Campos', [{'campo': n, 'min': v.min(), 'max': v.max(), 'media': v.mean()} for n, v in values.items()],
                       caption='Magnitud para los campos vectoriales; ultimo paso en un estudio transitorio.')

        self._add_views(model, values, plt)

        if getattr(model, 'convergence', None):
            self._add_convergence(model, plt)

        if getattr(model, 'probe_values', None) and model.probe_times:
            figure, ax = plt.subplots(figsize=(6.5, 3.2))
            for name, series in model.probe_values.items():
                series = np.asarray(series, dtype=float)
                ax.plot(model.probe_times, series if series.ndim == 1 else np.linalg.norm(series, axis=1), label=name)
            ax.set_xlabel('tiempo (o iteracion)'); ax.grid(alpha=0.3); ax.legend()
            self.add_figure('Sondas', figure)

    def _add_views(self, model, values, plt):

        mesh = model.mesh
        planar = len(mesh.NL[0].coor) == 2

        if planar:
            try:
                triang = mesh.triangulation(scale=1.0)
            except Exception as error:                       # an element type without triangulation
                self.add_text('Campos en la malla', f"No se pudo triangular la malla ({error}).")
                return

            n = len(values)
            columns = min(2, n)
            figure, axes = plt.subplots((n + columns - 1) // columns, columns, figsize=(6.2 * columns, 3.6 * ((n + columns - 1) // columns)),
                                        squeeze=False, constrained_layout=True)
            for ax, (name, array) in zip(axes.ravel(), values.items()):
                spread = array.max() - array.min()
                levels = 24 if spread > 1e-12 * max(abs(array).max(), 1e-300) else 1
                cs = ax.tricontourf(triang, array, levels=levels, cmap='viridis')
                figure.colorbar(cs, ax=ax)
                ax.set_aspect('equal'); ax.set_title(name)
            for ax in axes.ravel()[n:]:
                ax.axis('off')
            self.add_figure('Campos', figure, caption='r-z en un caso axisimetrico (x = r).' if model.axisymmetric else None)
            return

        from Source.Post_processing import Viewer
        if not Viewer.available():
            self.add_text('Campos en la malla', "Malla 3D: las vistas necesitan pyvista (pip install pyvista); las estadisticas de arriba "
                          "y Model.exportVTK (ParaView) cubren el resto.")
            return

        for name in values:
            with tempfile.TemporaryDirectory() as folder:
                png = os.path.join(folder, 'view.png')
                Viewer.show(model, name, screenshot=png)
                with open(png, 'rb') as file:
                    self.add_image(f"Campo {name}", file.read())

    def _add_convergence(self, model, plt):

        rows = model.convergence
        figure, ax = plt.subplots(figsize=(6.5, 3.2))
        if 'relative_residual' in rows[0] and 'step' not in rows[0]:
            ax.semilogy([r['iteration'] for r in rows], [max(r['relative_residual'], 1e-300) for r in rows], 'o-')
            ax.set_xlabel('iteracion'); ax.set_ylabel('||R|| / ||R0||'); title = 'Convergencia'
        else:
            ax.plot([r['time'] for r in rows], [r.get('newton_iterations', np.nan) for r in rows], '.-')
            ax.set_xlabel('tiempo'); ax.set_ylabel('iteraciones de Newton por paso'); title = 'Iteraciones por paso'
        ax.grid(alpha=0.3)
        self.add_figure(title, figure)

    # ----- output -----

    def html(self):

        sub = f'<div class="sub">{html.escape(self.subtitle)}</div>' if self.subtitle else ''
        stamp = time.strftime('%Y-%m-%d %H:%M')
        return (f'<!doctype html>\n<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{html.escape(self.title)}</title><style>{STYLE}</style></head><body>\n'
                f'<h1>{html.escape(self.title)}</h1>{sub}<div class="sub">FEM-sim · {stamp}</div>\n'
                + '\n'.join(self._parts + self._tail) + '\n</body></html>\n')

    def save(self, path):

        with open(path, 'w', encoding='utf-8') as file:
            file.write(self.html())
        return path


def case_report(title, params, results, subtitle=None):

    """A Report with the parameters (as JSON) and the scalar results of a case; add_model / add_figure then complete it."""

    report = Report(title, subtitle)
    report.add_table('Resultados', {k: v for k, v in results.items() if not isinstance(v, (list, dict, np.ndarray))})
    arrays = {k: v for k, v in results.items() if isinstance(v, (list, np.ndarray)) and np.ndim(v) == 1 and len(v) > 0
              and all(isinstance(x, (int, float)) for x in np.asarray(v).tolist())}
    if arrays:
        length = {len(v) for v in arrays.values()}
        if len(length) == 1:
            n = length.pop()
            report.add_table('Tabla de resultados', [{k: np.asarray(v)[i].item() for k, v in arrays.items()} for i in range(n)])
    report.add_code('Caso', json.dumps(params, indent=2, ensure_ascii=False, default=str), last=True)
    return report
