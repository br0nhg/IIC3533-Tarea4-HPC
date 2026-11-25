"""Tarea 4 (d): tiempos de NumPy y JAX en un computador propio (CPU, sin GPU).

Usa la misma grid, la misma referencia NumPy y la misma versión JAX que el notebook.
Guarda resultados/tiempos_cpu.csv (se actualiza después de cada medición) y
resultados/info_cpu.json.

Uso:
    python benchmark_cpu.py                      # W = 1024 2048 4096 8192, 3 repeticiones
    python benchmark_cpu.py --Ws 1024 2048 --reps 1
"""
import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

MAX_ITER = 100
TOL = 1e-6
ROOTS = np.array([1.0, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)], dtype=np.complex64)

REGIONS = {
    "clasica": (-2.0, 2.0, -2.0, 2.0),
    "frontera_central": (-0.3, 0.3, -0.3, 0.3),
    "frontera_lateral": (0.3, 0.9, 0.3, 0.9),
}


def grid(region, W):
    x_min, x_max, y_min, y_max = REGIONS[region]
    x = np.linspace(x_min, x_max, W, dtype=np.float32)
    y = np.linspace(y_min, y_max, W, dtype=np.float32)
    return (x[None, :] + 1j * y[:, None]).astype(np.complex64)


def newton_numpy(z0):
    z = z0.copy()
    count = np.zeros(z.shape, dtype=np.int32)
    active = np.ones(z.shape, dtype=bool)
    for _ in range(MAX_ITER):
        z_new = z[active] - (z[active] ** 3 - 1) / (3 * z[active] ** 2)
        converged = np.abs(z_new - z[active]) < TOL
        z[active] = z_new
        count[active] += 1
        active[active] = ~converged
        if not active.any():
            break
    root_idx = np.full(z.shape, -1, dtype=np.int32)
    for i, r in enumerate(ROOTS):
        root_idx[np.abs(z - r) < 1e-3] = i
    return root_idx, count


ROOTS_JAX = jnp.asarray(ROOTS)


def pixel(z0):
    def cond(estado):
        z, k, activo = estado
        return activo & (k < MAX_ITER)

    def body(estado):
        z, k, activo = estado
        z2 = z * z
        z_new = z - (z2 * z - 1) / (3 * z2)
        activo = ~(jnp.abs(z_new - z) < TOL)
        return z_new, k + 1, activo

    z, k, _ = jax.lax.while_loop(cond, body, (z0, jnp.int32(0), jnp.bool_(True)))
    cerca = jnp.abs(z - ROOTS_JAX) < 1e-3
    root_idx = jnp.where(jnp.any(cerca), jnp.argmax(cerca), -1).astype(jnp.int32)
    return root_idx, k


@jax.jit
def newton_jax(z0):
    return jax.vmap(jax.vmap(pixel))(z0)


def medir_numpy(z0):
    t = time.perf_counter()
    newton_numpy(z0)
    return time.perf_counter() - t


def medir_jax(z0):
    jax.block_until_ready(newton_jax(z0))
    t = time.perf_counter()
    jax.block_until_ready(newton_jax(z0))
    return time.perf_counter() - t


def mismatch_fraction(a, b):
    return float(np.mean(np.asarray(a) != np.asarray(b)))


def sysctl(nombre):
    try:
        return subprocess.check_output(["sysctl", "-n", nombre], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def info_sistema():
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "sistema": platform.platform(),
        "cpu_nucleos_logicos": os.cpu_count(),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    if platform.system() == "Darwin":
        info["cpu_modelo"] = sysctl("machdep.cpu.brand_string")
        info["cpu_nucleos_fisicos"] = int(sysctl("hw.physicalcpu"))
        perf, efic = sysctl("hw.perflevel0.physicalcpu"), sysctl("hw.perflevel1.physicalcpu")
        if perf and efic:
            info["cpu_nucleos_rendimiento"] = int(perf)
            info["cpu_nucleos_eficiencia"] = int(efic)
        info["ram_total_GB"] = round(int(sysctl("hw.memsize")) / 1024**3, 2)
    else:
        try:
            import psutil
            info["cpu_nucleos_fisicos"] = psutil.cpu_count(logical=False)
            info["ram_total_GB"] = round(psutil.virtual_memory().total / 1024**3, 2)
        except ImportError:
            pass
        info["cpu_modelo"] = platform.processor()
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ws", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--regiones", nargs="+", default=list(REGIONS))
    p.add_argument("--salida", default="resultados")
    args = p.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    info = info_sistema()
    with open(os.path.join(args.salida, "info_cpu.json"), "w") as f:
        json.dump(info, f, indent=2)
    for k, v in info.items():
        print(f"{k:26s} {v}")

    ruta = os.path.join(args.salida, "tiempos_cpu.csv")
    filas = []
    for W in args.Ws:
        for region in args.regiones:
            z0 = grid(region, W)
            r_ref, c_ref = newton_numpy(z0)
            r_jx, c_jx = (np.asarray(a) for a in newton_jax(jnp.asarray(z0)))
            mis_r, mis_c = mismatch_fraction(r_ref, r_jx), mismatch_fraction(c_ref, c_jx)
            for nombre, medir, entrada in [("NumPy", medir_numpy, z0), ("JAX", medir_jax, jnp.asarray(z0))]:
                ts = [medir(entrada) for _ in range(args.reps)]
                filas.append(dict(
                    region=region, W=W, version=nombre,
                    t_mediana=float(np.median(ts)), t_min=float(min(ts)), t_max=float(max(ts)),
                    referencia="NumPy", mismatch_root=0.0 if nombre == "NumPy" else mis_r,
                    mismatch_count=0.0 if nombre == "NumPy" else mis_c))
                print(f"{region:18s} W={W:5d} {nombre:6s} {filas[-1]['t_mediana']:9.3f} s"
                      + (f"   mismatch root={mis_r:.1e} count={mis_c:.1e}" if nombre == "JAX" else ""),
                      flush=True)
                pd.DataFrame(filas).to_csv(ruta, index=False)
    print("Resultados en", ruta)


if __name__ == "__main__":
    main()
