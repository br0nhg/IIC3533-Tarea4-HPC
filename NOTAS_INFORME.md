# Notas para el informe — Tarea 4

Este documento reúne los datos y los puntos de análisis para redactar el informe en LaTeX. Todos los archivos citados están en `resultados/`. Cada tabla está en `.csv` y en `.tex` (formato `booktabs`, requiere `\usepackage{booktabs}`), y cada figura en `.png` y en `.pdf`. En LaTeX conviene usar los `.pdf`, que son vectoriales.

## 1. Recursos utilizados

| | Google Colab | Computador propio |
|---|---|---|
| CPU | Intel Xeon @ 2.00 GHz, 2 vCPU (1 núcleo físico) | Apple M5, 10 núcleos (4 de rendimiento + 6 de eficiencia) |
| RAM | 12,67 GB | 16 GB |
| GPU | NVIDIA Tesla T4: 40 SMs, 14,56 GB, compute capability 7.5, 1590 MHz, warp de 32 | — |
| Software | Python 3.13, NumPy 2.4.6, CuPy 14.0.1, JAX 0.11.1, CUDA 13.0 | Python 3.12, NumPy 2.5.5, JAX 0.12.1 (CPU) |

Las fuentes son `info_colab.json` e `info_cpu.json`. El notebook también muestra la salida de `nvidia-smi` y `lscpu`.

## 2. Metodología

- Parámetros: `MAX_ITER = 100` y `TOL = 1e-6`, en precisión simple (complex64/float32).
- Tiempos:
  - CuPy: eventos CUDA.
  - JAX: `block_until_ready`.
  - NumPy: `perf_counter`.
  - Cada medición descarta una llamada de calentamiento; en JAX, esa llamada incluye la compilación.
  - Se reporta la **mediana de 3 repeticiones** (`t_mediana`, junto con `t_min` y `t_max`).
- Los tiempos de RawKernel incluyen la separación de `z0` en `x`/`y` y la reserva de memoria de las salidas, porque eso ocurre dentro de `newton_rawkernel`. No incluyen las transferencias CPU↔GPU, porque `z0` ya está en la GPU. Lo mismo vale para CuPy vectorizado y JAX.
- Verificación: se calcula `mismatch_fraction` de `root_idx`.
  - Con $W = 1024$ se compara contra NumPy.
  - Con $W > 1024$ se compara contra CuPy vectorizado, que replica la lógica de NumPy.
- Detalle de implementación: en CuPy, RawKernel y JAX, $z^3$ se calcula como $z^2 \cdot z$. NumPy también usa multiplicaciones para potencias enteras complejas, mientras que el `**` complejo de CuPy usa exp/log y se aleja más de la referencia.

## 3. Mapa de preguntas → material

| Pregunta | Material |
|---|---|
| (a) CuPy vectorizado | Notebook, sección (a): `mismatch fraction = 3,4e-5` y `numpy 0,445 s / cupy 0,108 s` (clásica, $W=1024$) |
| (b) RawKernel | Notebook, sección (b): `mismatch = 3,2e-5`, `0,4 ms` |
| (c) JAX | Notebook, sección (c): `mismatch = 1,9e-5`, `8,8 ms` |
| (d) Imágenes | `fractales_W1024.pdf` |
| (d) Tiempos GPU | `tabla_tiempos_gpu.tex`, `tiempos_gpu.pdf`, `rendimiento_gpu.pdf` (Mpixeles/s) |
| (d) CPU vs GPU | `tabla_cpu_gpu.tex`, `tabla_speedup_cpu_gpu.tex`, `tiempos_cpu_vs_gpu.pdf` |
| (e) Divergencia | `estadisticas_iteraciones.tex`, `divergencia_W1024.pdf`, `histograma_iteraciones.pdf` |
| (e) JAX y tiempo vs trabajo | `iteraciones_jax.tex`, `tiempo_vs_trabajo.pdf` |

`tabla_cpu_gpu`, `tabla_speedup_cpu_gpu`, `iteraciones_jax` y `tiempo_vs_trabajo` se generan en local con `analisis.py`.

## 4. Resultados clave para (d)

### Tiempos (mediana, segundos)

| Región | W | NumPy CPU | JAX CPU | CuPy vect. GPU | RawKernel GPU | JAX GPU |
|---|---|---|---|---|---|---|
| Clásica | 1024 | 0,110 | 0,157 | 0,0889 | 0,000289 | 0,00878 |
| Clásica | 8192 | 9,14 | 6,69 | 5,82 | 0,0218 | 0,587 |
| F. central | 1024 | 0,147 | 0,176 | 0,0948 | 0,000356 | 0,00943 |
| F. central | 8192 | 11,54 | 8,20 | 6,25 | 0,0283 | 0,636 |
| F. lateral | 1024 | 0,128 | 0,158 | 0,0936 | 0,000311 | 0,00914 |
| F. lateral | 8192 | 8,81 | 7,48 | 6,19 | 0,0243 | 0,671 |

La tabla completa, con los cuatro valores de W, está en `tabla_cpu_gpu.tex`.

### Comparación entre versiones en GPU (mismo W y región)

- **RawKernel es la versión más rápida por lejos:**
  - 220–308× más rápido que CuPy vectorizado;
  - 22–39× más rápido que JAX en GPU;
  - procesa unos 2.400–3.600 Mpixeles/s.
- **JAX en GPU es 7–10× más rápido que CuPy vectorizado.**
- **CuPy vectorizado es la versión GPU más lenta.** Cada iteración:
  - lanza muchos kernels pequeños: indexación booleana `z[active]`, potencias, división, `abs`, comparaciones, asignación con máscara;
  - hace indexación booleana, que requiere un *scan* para compactar;
  - llama a `active.any()`, que obliga a sincronizar la GPU con la CPU.

  Además, las máscaras `active` y las asignaciones `z[active] = ...` recorren arreglos de tamaño $W^2$ en **todas** las iteraciones, aunque queden pocos pixeles activos. Así, el costo por iteración no baja mucho a medida que los pixeles convergen.
  - Con $W=1024$ son unos 1,5 ms por iteración. A ese tamaño domina el overhead de lanzamiento y sincronización.
  - Con $W=8192$ son unos 90 ms por iteración. Ahí domina el tráfico de memoria.

### Escalamiento con W

Pasar de $W=1024$ a $8192$ multiplica los pixeles por 64. Los tiempos crecen:

| Versión | Factor |
|---|---|
| CuPy vectorizado | ≈ 64–66× |
| JAX GPU | ≈ 67–73× |
| RawKernel | ≈ 75–79× |
| NumPy CPU | ≈ 69–83× |
| JAX CPU | ≈ 43–47× |

Todas las versiones escalan aproximadamente como $O(W^2)$.

- **RawKernel con W pequeño:** a $W=1024$ hay solo $2^{20}$ threads (4096 bloques), y el tiempo (≈0,3 ms) incluye overheads fijos: reserva de memoria, copia de `x`/`y` y lanzamiento.
- **JAX en CPU** escala mejor que lineal entre $W=2048$ y $4096$. No investigamos la causa, así que no conviene afirmar nada concreto sobre esto.

### CPU vs GPU

| Comparación | Speedup |
|---|---|
| RawKernel (T4) vs NumPy en el M5 | 360–500× |
| RawKernel (T4) vs JAX CPU en el M5 | 290–680× |
| JAX GPU vs JAX CPU (mismo código) | 9–20× |
| CuPy vectorizado (GPU) vs NumPy en el M5 | solo 1,2–2,0× |

- **CuPy vectorizado:** mover a la GPU la lógica de máscaras de NumPy casi no sirve, por los motivos explicados en la sección anterior.
- **CPU de Colab vs Mac:** el NumPy de referencia en la CPU de Colab, con $W=1024$, tarda 0,46–0,66 s, unas 4× más que en el M5. La CPU de Colab es débil (2 vCPU); esto conviene comentarlo al comparar.
- **JAX CPU vs NumPy en el M5:**
  - Con W pequeño, JAX es más lento.
  - Con $W \geq 4096$, JAX es 1,2–1,5× más rápido.
  - JAX itera *todos* los pixeles hasta el máximo global (ver (e)), pero lo hace en un loop compilado y fusionado. NumPy reduce la aritmética a los pixeles activos, pero en cada iteración crea arreglos temporales y recorre máscaras completas.

## 5. Resultados clave para (e)

### Métricas de divergencia

Las fuentes son `estadisticas_iteraciones.csv`, calculado con los `count` del RawKernel, y `analisis.py`.

- **Warp:** con bloques 1D de 256 threads sobre la grid aplanada por filas, un warp son 32 pixeles consecutivos de una misma fila.
- **`iter_max_warp_media`:** promedio, sobre todos los warps, del máximo de `count` dentro del warp. Es lo que realmente "paga" cada thread del warp.
- **`eficiencia_warp`** $= \sum_p \text{count}_p / \sum_w 32 \max_{p\in w} \text{count}_p$. Vale 1 cuando no hay divergencia.

| Región | W | iter. media | desv. | máx. por warp (media) | eficiencia warp | iteraciones "perdidas" por pixel |
|---|---|---|---|---|---|---|
| Clásica | 1024 | 7,84 | 3,11 | 9,60 | 0,817 | 1,75 |
| F. central | 1024 | 12,22 | 3,71 | 14,24 | 0,858 | 2,02 |
| F. lateral | 1024 | 9,68 | 3,72 | 11,47 | 0,844 | 1,79 |
| Clásica | 8192 | 7,84 | 3,11 | 8,39 | 0,935 | 0,55 |
| F. central | 8192 | 12,22 | 3,71 | 12,87 | 0,950 | 0,65 |
| F. lateral | 8192 | 9,69 | 3,72 | 10,23 | 0,946 | 0,55 |

### Puntos de análisis para RawKernel

1. **Expectativa a priori: la frontera central.**
   - En el origen, $f'(0)=0$, así que el paso de Newton explota. Además, ahí se tocan las tres cuencas, por lo que la región está llena de frontera fractal.
   - Pixeles vecinos pueden necesitar números de iteraciones muy distintos. Esa región tiene la mayor iteración media (12,2) y la mayor dispersión.
   - La frontera lateral también está casi toda cubierta de frontera. La clásica tiene grandes zonas "lisas" cerca de cada raíz, que convergen en pocas iteraciones.
2. **Lo que muestran los datos.**
   - **En términos absolutos**, la frontera central es la que más iteraciones desperdicia por pixel (2,02 con $W=1024$, contra 1,75–1,79 en las otras regiones). También es la de mayor `iter_max_warp_media`.
   - **En términos relativos**, la clásica tiene la *menor* eficiencia de warp con $W=1024$ (0,817). Al cubrir un área de 4×4, cada warp abarca un tramo mayor del plano ($32 \cdot 4/1024 = 0,125$, contra $0,019$ en las regiones de 0,6×0,6). Por eso es más probable que un warp cruce una frontera.
   - La divergencia relativa es parecida en las tres regiones (eficiencia entre 0,82 y 0,95).
   - **Interpretación sugerida:** la diferencia de tiempo entre regiones se explica sobre todo por cuántas iteraciones hay que hacer (la iteración media), y en menor medida por la divergencia.
3. **La divergencia disminuye al aumentar W** (eficiencia de 0,82–0,86 con $W=1024$ a 0,93–0,95 con $W=8192$). Con una grid más fina, 32 pixeles consecutivos están más cerca entre sí y su número de iteraciones es más parecido. Solo los warps que tocan la frontera fractal siguen divergiendo.
4. **¿Son consistentes los tiempos?** Sí.
   - Para todo W, el orden de los tiempos del RawKernel es central > lateral > clásica (8192: 28,3 / 24,3 / 21,8 ms). Es el mismo orden que `iter_max_warp_media` (12,9 / 10,2 / 8,4).
   - Normalizando por pixel y por iteración efectiva del warp, el costo es casi constante: unos 0,023–0,029 ns con $W \le 4096$ y 0,033–0,039 ns con $W=8192$ (`iteraciones_jax.csv`, columna `ns_por_pixel_iter_rk`; figura `tiempo_vs_trabajo.pdf`, izquierda).
   - La razón de tiempos central/clásica (1,30 con $W=8192$) es menor que la razón de iteraciones (1,53), porque una parte del tiempo es fija por pixel: lectura de `x`/`y`, escritura de resultados, reservas y copias dentro de la función medida.

### Puntos de análisis para JAX: `vmap` + `while_loop`

1. **Lo que dice la documentación.**
   - La documentación de `jax.lax.while_loop` solo describe la semántica secuencial, y la de `jax.vmap` no menciona el control de flujo.
   - La de `jax.lax.cond` sí explica que, con `vmap` sobre un predicado vectorizado, `cond` se convierte en `select` y se ejecutan ambas ramas: <https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html>.
   - Para `while_loop` el comportamiento es análogo. Lo verificamos en el código fuente de JAX 0.11.1, la misma versión de Colab, en el archivo `jax/_src/lax/control_flow/loops.py`:
     - `_while_loop_batching_rule`: "If the predicate is batched, we have to batch *all* of the carry".
     - `_while_lowering`: la condición se reduce con `lax.reduce_or`, y el cuerpo actualiza el estado con un `select` (`_pred_bcast_select_hlo`).
   - Se comprueba con `jax.jit(jax.vmap(pixel)).lower(...).as_text()`, que muestra un `stablehlo.while` cuya condición es un `reduce ... or` y cuyo cuerpo termina en `stablehlo.select`.
2. **Respuesta.**
   - El loop vectorizado sigue mientras **algún** pixel siga activo.
   - En cada vuelta, el cuerpo se calcula para **todos** los pixeles, y los que ya terminaron descartan el resultado gracias al `select`.
   - Entonces, un pixel que converge en pocas iteraciones ejecuta igual **$\max_p \text{count}_p$ iteraciones**, el máximo sobre *toda* la grid, y no solo sobre su warp.
3. **Números.** El máximo global de iteraciones se recalculó con la misma función JAX en CPU (`analisis.py`).

   | Región | W | máximo global | media | eficiencia (media/máx.) |
   |---|---|---|---|---|
   | Clásica | 1024 | 54 | 7,84 | 0,145 |
   | Clásica | 8192 | 69 | 7,84 | 0,114 |
   | F. central | 1024 | 57 | 12,22 | 0,214 |
   | F. central | 8192 | 82 | 12,22 | 0,149 |
   | F. lateral | 1024 | 54 | 9,68 | 0,179 |
   | F. lateral | 8192 | 66 | 9,69 | 0,147 |

   - El máximo global varía entre 54 y 82, mientras que el promedio es de 8 a 12. Por lo tanto, solo un **11–21 % del trabajo de JAX es útil**.
   - El máximo lo alcanzan apenas 1 a 4 pixeles de toda la grid (columna `pixeles_en_max_jax`).
   - Con W más grande aparecen pixeles más "difíciles" cerca de la frontera fractal, así que el máximo global crece.
4. **Relación con los tiempos.**
   - El costo de JAX por pixel y por iteración del loop es casi constante: 0,11–0,16 ns (`ns_por_pixel_iter_jax`; `tiempo_vs_trabajo.pdf`, derecha). El tiempo de JAX es, entonces, proporcional a $W^2 \cdot \max \text{count}$.
   - La diferencia JAX/RawKernel (≈30× con $W=1024$ en la clásica) se descompone en dos factores:
     - **Iteraciones ejecutadas:** máximo global contra máximo por warp, $54/9,6 \approx 5,6\times$.
     - **Costo por iteración:** $0,163/0,029 \approx 5,7\times$.
     - El producto, ≈ 32×, coincide con lo medido (30,4×).
   - **Explicación probable del costo por iteración** (no medida directamente con un profiler): en el `while` de XLA, cada vuelta es al menos un kernel que lee y escribe el estado completo (z, count, activo) en memoria global. Además, la condición necesita una reducción global y su resultado vuelve a la CPU para decidir si sigue el loop. En el RawKernel, en cambio, el estado vive en registros durante todo el loop.

## 6. Diferencias con la referencia (precisión)

- En todas las versiones `mismatch_fraction(root_idx)` está entre $2\cdot10^{-5}$ y $8\cdot10^{-5}$: unos pocos pixeles por millón, siempre sobre la frontera fractal.
- En la frontera, un error de redondeo en float32 (distinto orden de operaciones, FMA, algoritmo de división compleja) cambia la cuenca a la que llega el pixel.
- `mismatch` de `count` ≈ $10^{-3}$: una iteración de diferencia en pixeles cuya diferencia $|z_{n+1}-z_n|$ queda muy cerca de `TOL`.
- **Detalle del RawKernel.** En unos pocos pixeles con $|z_0|$ muy pequeño (por ejemplo $z_0 = \pm3,66\cdot10^{-5} \pm 3,66\cdot10^{-5}i$, en la frontera central con $W=8192$), el primer paso lanza $z$ a $\sim10^8$.
  - En el paso siguiente, la división compleja "ingenua" del kernel ($(a+bi)/(c+di)$ con $c^2+d^2$) desborda float32 y produce NaN.
  - Esos pixeles quedan con `root_idx = -1` y `count = 100`. Son 4 pixeles de 67 millones con $W=8192$ y 1 de 16 millones en la lateral con $W=4096$.
  - NumPy y JAX usan una división escalada que evita el desborde.
  - Por eso `iter_max = 100` aparece en `estadisticas_iteraciones.csv` en esos casos. No afecta el análisis, que usa el máximo por warp o el máximo recalculado de JAX.
  - Se puede mencionar como una limitación, o corregirlo en el kernel calculando el paso como $\tfrac{1}{3}(z - 1/z^2)$ con $1/z = \bar z/|z|^2$.

## 7. Declaración de uso de IA (borrador)

> En esta tarea se utilizó Claude (Anthropic), a través de Claude Code, como asistente para:
> - implementar las versiones CuPy vectorizada, CuPy RawKernel y JAX;
> - escribir el código de medición, tablas y figuras;
> - escribir el *script* de medición en CPU;
> - preparar un primer análisis de los resultados.
>
> El equipo revisó el código, ejecutó el *notebook* en Google Colab y redactó el informe, verificando las conclusiones con los datos medidos.

Ajustar este texto a cómo lo usó realmente el equipo.
