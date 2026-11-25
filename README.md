# IIC3533 · Tarea 4 · Fractal de Newton en GPU

Implementaciones del fractal de Newton para $f(z) = z^3 - 1$ en tres versiones:

- CuPy vectorizado
- CuPy RawKernel
- JAX (`lax.while_loop` + `vmap` + `jit`)

Se miden en GPU (Google Colab, Tesla T4) y se comparan con NumPy y JAX en CPU (Apple M5).

Plazo de entrega: **jueves 3 de diciembre de 2026, 23:59**.

## Estructura

```
├── enunciado/
│   ├── tarea04.pdf               Enunciado
│   └── tarea04_template.ipynb    Template original del curso, sin modificar
├── entrega/
│   └── tarea04_completo.ipynb    Notebook completo, ejecutado en Colab (se entrega en Canvas)
├── scripts/
│   ├── benchmark_cpu.py          Tiempos de NumPy y JAX en un computador propio (pregunta d)
│   └── analisis.py               Comparación CPU vs GPU e iteraciones de JAX (preguntas d y e)
├── informe/
│   └── NOTAS_INFORME.md          Números clave y puntos de análisis para redactar el informe
└── resultados/
    ├── figuras/                  Figuras .pdf (para LaTeX) y .png
    ├── tablas/                   Tablas .tex (booktabs)
    └── datos/                    CSV con tiempos y estadísticas, e información del hardware (.json)
```

## Entrega en Canvas

- El PDF del informe, que redacta el equipo a partir de `informe/NOTAS_INFORME.md` y `resultados/`.
- `entrega/tarea04_completo.ipynb`.

## Cómo reproducir

### GPU (Google Colab)

1. Abrir `entrega/tarea04_completo.ipynb` en Colab y elegir *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.
2. *Entorno de ejecución → Ejecutar todo*.
3. La última celda descarga `resultados_tarea04.zip`. En Colab, todos los resultados quedan juntos en una sola carpeta `resultados/`. En este repo se repartieron en `figuras/`, `tablas/` y `datos/`.

### CPU (computador propio)

Desde la raíz del repo:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy jax pandas psutil matplotlib jinja2
.venv/bin/python scripts/benchmark_cpu.py     # W = 1024..8192, 3 repeticiones -> resultados/datos/
.venv/bin/python scripts/analisis.py          # tablas y figuras CPU vs GPU
```

Opciones de `benchmark_cpu.py`: `--Ws 1024 2048`, `--reps 1` y `--regiones clasica`.

## Verificación

Cada versión compara su `root_idx` con la referencia mediante `mismatch_fraction`:

- con NumPy para $W = 1024$;
- con CuPy vectorizado para $W > 1024$.

Las diferencias, del orden de $10^{-5}$ a $10^{-4}$, vienen del redondeo en precisión simple sobre la frontera fractal.

## Uso de IA

Parte del código y del análisis se preparó con ayuda de Claude (Anthropic), lo que debe declararse en el informe.
