"""Análisis posterior para el informe: combina los tiempos de GPU (Colab) y CPU (propio).

Entradas (en resultados/):
    tiempos_gpu.csv, tiempos_cpu.csv, estadisticas_iteraciones.csv

Salidas (en resultados/):
    tabla_cpu_gpu.csv/.tex        tiempos de todas las versiones
    tabla_speedup_cpu_gpu.csv/.tex  speedups de GPU respecto de CPU
    iteraciones_jax.csv/.tex      máximo global de iteraciones (lo que ejecuta vmap+while_loop)
                                  y tiempo por iteración de JAX en GPU
    tiempos_cpu_vs_gpu.png/.pdf
    tiempo_vs_trabajo.png/.pdf    tiempo por pixel vs iteraciones efectivas (pregunta e)

El máximo global de iteraciones se recalcula en CPU con la misma función JAX del notebook,
porque las estadísticas de Colab se tomaron de los count del RawKernel.

Uso:  python analisis.py [--sin-recalcular]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "resultados"
REGIONES = ["clasica", "frontera_central", "frontera_lateral"]
TITULOS = {"clasica": "Clásica", "frontera_central": "Frontera central",
           "frontera_lateral": "Frontera lateral"}
Ws = [1024, 2048, 4096, 8192]

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
})


def guardar(fig, nombre):
    fig.savefig(os.path.join(R, nombre + ".png"), dpi=200)
    fig.savefig(os.path.join(R, nombre + ".pdf"))
    plt.close(fig)


def a_latex(df, nombre, fmt):
    with open(os.path.join(R, nombre + ".tex"), "w") as f:
        f.write(df.to_latex(float_format=fmt, na_rep="--"))
    df.to_csv(os.path.join(R, nombre + ".csv"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sin-recalcular", action="store_true",
                   help="no recalcula el máximo de iteraciones con JAX/NumPy en CPU")
    args = p.parse_args()

    gpu = pd.read_csv(os.path.join(R, "tiempos_gpu.csv"))
    cpu = pd.read_csv(os.path.join(R, "tiempos_cpu.csv"))
    stats = pd.read_csv(os.path.join(R, "estadisticas_iteraciones.csv")).set_index(["region", "W"])

    gpu = gpu[gpu.version != "NumPy (CPU Colab)"].copy()
    gpu["version"] = gpu["version"] + " (GPU)"
    cpu["version"] = cpu["version"] + " (CPU propio)"
    todos = pd.concat([gpu, cpu], ignore_index=True)

    # ------------------------------------------------------------ tabla de tiempos
    orden = ["NumPy (CPU propio)", "JAX (CPU propio)", "CuPy vectorizado (GPU)",
             "CuPy RawKernel (GPU)", "JAX (GPU)"]
    tabla = todos.pivot_table(index=["region", "W"], columns="version",
                              values="t_mediana").reindex(columns=orden)
    a_latex(tabla, "tabla_cpu_gpu", "%.4g")

    sp = pd.DataFrame(index=tabla.index)
    sp["RawKernel vs NumPy CPU"] = tabla["NumPy (CPU propio)"] / tabla["CuPy RawKernel (GPU)"]
    sp["RawKernel vs JAX CPU"] = tabla["JAX (CPU propio)"] / tabla["CuPy RawKernel (GPU)"]
    sp["JAX GPU vs JAX CPU"] = tabla["JAX (CPU propio)"] / tabla["JAX (GPU)"]
    sp["CuPy vect. vs NumPy CPU"] = tabla["NumPy (CPU propio)"] / tabla["CuPy vectorizado (GPU)"]
    a_latex(sp, "tabla_speedup_cpu_gpu", "%.1f")
    print(tabla.to_string(float_format=lambda v: f"{v:.4g}"))
    print(sp.to_string(float_format=lambda v: f"{v:.1f}"))

    # ------------------------------------------------------------ figura CPU vs GPU
    estilos = {
        "CuPy vectorizado (GPU)": dict(color="#1f77b4", marker="o"),
        "CuPy RawKernel (GPU)": dict(color="#d62728", marker="s"),
        "JAX (GPU)": dict(color="#2ca02c", marker="^"),
        "NumPy (CPU propio)": dict(color="#7f7f7f", marker="D", ls="--"),
        "JAX (CPU propio)": dict(color="#9467bd", marker="v", ls="--"),
    }
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True, constrained_layout=True)
    for ax, region in zip(axs, REGIONES):
        for version, estilo in estilos.items():
            d = todos[(todos.region == region) & (todos.version == version)].sort_values("W")
            ax.plot(d.W, d.t_mediana, label=version, lw=2, ms=7, **estilo)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Ws, [str(w) for w in Ws])
        ax.set_title(TITULOS[region])
        ax.set_xlabel("$W$ (grid de $W \\times W$ pixeles)")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel("Tiempo [s]")
    axs[0].legend(fontsize=10)
    fig.suptitle("Tiempo de ejecución: CPU propio (Apple M5) vs GPU de Colab (Tesla T4)")
    guardar(fig, "tiempos_cpu_vs_gpu")

    # ------------------------------------------------------------ iteraciones de JAX
    ruta_it = os.path.join(R, "iteraciones_jax.csv")
    if args.sin_recalcular and os.path.exists(ruta_it):
        it = pd.read_csv(ruta_it).set_index(["region", "W"])
    else:
        import jax.numpy as jnp
        from benchmark_cpu import grid, newton_numpy, newton_jax
        filas = []
        for region in REGIONES:
            for W in Ws:
                z0 = grid(region, W)
                _, c_np = newton_numpy(z0)
                _, c_jx = newton_jax(jnp.asarray(z0))
                c_jx = np.asarray(c_jx)
                filas.append(dict(region=region, W=W,
                                  iter_max_numpy=int(c_np.max()), iter_max_jax=int(c_jx.max()),
                                  pixeles_en_max_jax=int((c_jx == c_jx.max()).sum()),
                                  iter_media_jax=float(c_jx.mean())))
                print(filas[-1], flush=True)
        it = pd.DataFrame(filas).set_index(["region", "W"])

    t_jax = gpu[gpu.version == "JAX (GPU)"].set_index(["region", "W"])["t_mediana"]
    t_rk = gpu[gpu.version == "CuPy RawKernel (GPU)"].set_index(["region", "W"])["t_mediana"]
    W2 = np.array([w for _, w in it.index], dtype=float) ** 2
    it["t_jax_gpu_s"] = t_jax.reindex(it.index).values
    it["ms_por_iteracion_jax"] = it["t_jax_gpu_s"] / it["iter_max_jax"] * 1e3
    it["ns_por_pixel_iter_jax"] = it["t_jax_gpu_s"] / (W2 * it["iter_max_jax"]) * 1e9
    it["eficiencia_jax"] = it["iter_media_jax"] / it["iter_max_jax"]
    it["t_rawkernel_gpu_s"] = t_rk.reindex(it.index).values
    it["iter_max_warp_media_rk"] = stats["iter_max_warp_media"].reindex(it.index).values
    it["ns_por_pixel_iter_rk"] = it["t_rawkernel_gpu_s"] / (W2 * it["iter_max_warp_media_rk"]) * 1e9
    a_latex(it, "iteraciones_jax", "%.4g")
    print(it.to_string(float_format=lambda v: f"{v:.4g}"))

    # ------------------------------------------------------------ tiempo vs trabajo
    marcadores = {1024: "o", 2048: "s", 4096: "^", 8192: "D"}
    colores = {"clasica": "#1f77b4", "frontera_central": "#d62728", "frontera_lateral": "#2ca02c"}
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    for (region, W), fila in it.iterrows():
        kw = dict(color=colores[region], marker=marcadores[W], s=70)
        axs[0].scatter(fila.iter_max_warp_media_rk, fila.t_rawkernel_gpu_s / W**2 * 1e9, **kw)
        axs[1].scatter(fila.iter_max_jax, fila.t_jax_gpu_s / W**2 * 1e9, **kw)
    axs[0].set_xlabel("Media del máximo de iteraciones por warp")
    axs[0].set_ylabel("Tiempo por pixel [ns]")
    axs[0].set_title("CuPy RawKernel (GPU)")
    axs[1].set_xlabel("Máximo global de iteraciones en la grid")
    axs[1].set_ylabel("Tiempo por pixel [ns]")
    axs[1].set_title("JAX (GPU): vmap + while_loop")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker="o", ls="", ms=8, label=TITULOS[r]) for r, c in colores.items()]
    h += [Line2D([], [], color="k", marker=m, ls="", ms=8, label=f"$W={w}$") for w, m in marcadores.items()]
    for ax in axs:
        ax.grid(True, alpha=0.3)
    axs[1].legend(handles=h, loc="upper left", fontsize=10)
    fig.suptitle("Tiempo por pixel en función de las iteraciones que realmente se ejecutan")
    guardar(fig, "tiempo_vs_trabajo")


if __name__ == "__main__":
    main()
