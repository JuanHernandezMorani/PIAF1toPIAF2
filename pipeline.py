import subprocess
import sys
import os
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).resolve().parent
TOOL_DIR = ROOT_DIR / "tool"
TRANSFORM_SCRIPT = ROOT_DIR / "transform.py"
BOXCHECK_SCRIPT = ROOT_DIR / "boxCheck.py"


def run_command(cmd, cwd=None, debug=False):
    """Ejecuta un comando y espera a que termine."""
    try:
        print(f"\n🚀 Ejecutando: {' '.join(cmd)} (cwd={cwd})")
        if debug:
            # Muestra salida en vivo
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            # Solo muestra cuando termina
            subprocess.run(cmd, cwd=cwd, check=True, text=True)
        print(f"✅ Finalizado: {' '.join(cmd)}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando {cmd}: {e}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="Pipeline automático")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Muestra salida completa de cada comando en vivo",
    )
    args = parser.parse_args()

    if not TRANSFORM_SCRIPT.exists():
        print(f"❌ No se encontró {TRANSFORM_SCRIPT}")
        sys.exit(1)

    print("\n🔄 Ejecutando transform.py...")
    run_command([sys.executable, str(TRANSFORM_SCRIPT)], cwd=ROOT_DIR, debug=args.debug)

    if not BOXCHECK_SCRIPT.exists():
        print(f"❌ No se encontró {BOXCHECK_SCRIPT}")
        sys.exit(1)

    print("\n📦 Ejecutando boxCheck.py...")
    run_command([sys.executable, str(BOXCHECK_SCRIPT)], cwd=ROOT_DIR, debug=args.debug)

    print("\n🎉 Pipeline completo.")


if __name__ == "__main__":
    main()
