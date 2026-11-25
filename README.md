# IIC3533 · Tarea 4 · Fractal de Newton en GPU

Implementaciones del fractal de Newton para $f(z) = z^3 - 1$ en tres versiones:

- CuPy vectorizado
- CuPy RawKernel
- JAX (`lax.while_loop` + `vmap` + `jit`)

Se miden en GPU (Google Colab) y se comparan con NumPy y JAX en CPU.

Plazo de entrega: **jueves 3 de diciembre de 2026, 23:59**.

## Contenido

| Archivo | Descripción |
|---|---|
| `tarea04.pdf` | Enunciado |
| `tarea04_template.ipynb` | Template original del curso, sin modificar |
| `tarea04_completo.ipynb` | Notebook completo: preguntas (a)–(e), mediciones y figuras. Es el que se entrega |
| `benchmark_cpu.py` | Tiempos de NumPy y JAX en un computador propio (CPU), pregunta (d) |
| `analisis.py` | Combina los tiempos de CPU y GPU y recalcula el máximo de iteraciones de JAX. Genera tablas y figuras adicionales en `resultados/` |
| `resultados/` | CSV de tiempos, estadísticas de iteraciones, tablas `.tex`, figuras `.png`/`.pdf` e información del hardware |
| `NOTAS_INFORME.md` | Números clave y puntos de análisis para redactar el informe en LaTeX |

## Cómo reproducir

### GPU (Google Colab)

1. Abrir `tarea04_completo.ipynb` en Colab y elegir *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.
2. *Entorno de ejecución → Ejecutar todo*.
3. La última celda descarga `resultados_tarea04.zip`, con todo el contenido de `resultados/`.
4. Opcional: subir `resultados/tiempos_cpu.csv` a la carpeta `resultados/` de Colab antes de la sección "Comparación CPU vs GPU" para obtener ese gráfico dentro del notebook.

### CPU (computador propio)

```bash
python3 -m venv .venv
.venv/bin/pip install numpy jax pandas psutil
.venv/bin/python benchmark_cpu.py --salida resultados   # W = 1024..8192, 3 repeticiones
```

Opciones: `--Ws 1024 2048`, `--reps 1` y `--regiones clasica`.

## Verificación

Cada versión compara su `root_idx` con la referencia mediante `mismatch_fraction`:

- con NumPy para $W = 1024$;
- con CuPy vectorizado para $W > 1024$.

Las diferencias, del orden de $10^{-5}$ a $10^{-4}$, vienen del redondeo en precisión simple sobre la frontera fractal.

## Uso de IA

Parte del código y del análisis se preparó con ayuda de Claude (Anthropic), lo que debe declararse en el informe.
