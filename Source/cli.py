"""Command line of femsim:  python -m femsim caso.toml [--set seccion.clave=valor ...] [--output DIR] [--jobs N] [--dry-run] [--quiet]
(see Source/Cases.py for the case files)."""

import argparse
import sys


def main(argv=None):

    parser = argparse.ArgumentParser(prog='femsim', description='Resuelve un caso descrito en un archivo TOML.')
    parser.add_argument('case', nargs='?', help='archivo TOML del caso')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='SECCION.CLAVE=VALOR',
                        help='cambia un valor del caso (se puede repetir), por ejemplo --set process.flow_rate=2e-4')
    parser.add_argument('--output', '-o', help='carpeta de resultados (sustituye a [output].directory)')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='procesos para los puntos de un [sweep]')
    parser.add_argument('--dry-run', action='store_true', help='comprueba el archivo y lista las ejecuciones sin resolver')
    parser.add_argument('--quiet', '-q', action='store_true', help='sin mensajes de progreso en la consola')
    parser.add_argument('--list-types', action='store_true', help='lista los tipos de caso y sale')
    args = parser.parse_args(argv)

    from Source.logger_config import configure_logging
    from Source import Cases

    if args.list_types:
        for name, cls in sorted(Cases.CASE_TYPES.items()):
            print(f"{name}: {(cls.__doc__ or '').strip().splitlines()[0]}")
        return 0

    if not args.case:
        parser.error("indique el archivo del caso (o --list-types)")

    if args.quiet:
        configure_logging(console_level='WARNING')

    try:
        results = Cases.run_case(args.case, overrides=args.overrides, output=args.output, jobs=args.jobs, dry_run=args.dry_run)
    except (ValueError, FileNotFoundError) as error:
        print(f"femsim: {error}", file=sys.stderr)
        return 2

    for record in results:
        label = ', '.join(f"{key} = {value}" for key, value in record['sweep'].items()) or '-'
        if args.dry_run:
            print(f"{record['name']}  [{label}]")
        else:
            summary = ', '.join(f"{key} = {value:.6g}" if isinstance(value, float) else f"{key} = {value}"
                                for key, value in record['results'].items() if not isinstance(value, (list, dict)))
            print(f"{record['run']}  [{label}]: {summary}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
