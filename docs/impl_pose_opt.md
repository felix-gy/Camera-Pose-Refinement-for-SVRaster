# Registro de Implementación: Optimización de Poses de Cámara en SVRaster (Vía A CUDA)

Este documento registra en detalle la arquitectura matemática, modificaciones de código, integración en el flujo de entrenamiento y guías de uso para la optimización conjunta de las poses de cámara (6-DoF en el álgebra de Lie $\mathfrak{se}(3)$) y el campo de radiancia de vóxeles dispersos (SVRaster).

---

## 1. Visión General y Objetivos

En el pipeline estándar de SVRaster, los extrínsecos de la cámara ($c2w \in \mathbb{R}^{4 \times 4}$) se asumen fijos e inmutables, provenientes de COLMAP o de sintéticos. En presencia de ruido en las poses iniciales o cuando se requiere optimización desde cero, es imprescindible permitir que los gradientes de la función de pérdida fotométrica retropropaguen hacia los parámetros de pose de cada cámara de entrenamiento.

### Características Principales Implementadas:
1. **Derivación Analítica en CUDA (Vía A)**: Cálculo directo de $\frac{\partial \mathcal{L}}{\partial \mathbf{R}}$ y $\frac{\partial \mathcal{L}}{\partial \mathbf{t}}$ dentro del kernel de backward raymarching volumétrico de SVRaster (`backward.cu`), reduciendo en memoria compartida y acumulando atómicamente en un tensor `dL_dc2w`.
2. **Parametrización en $\mathfrak{se}(3)$ (Álgebra de Lie)**: Representación de perturbaciones 6-DoF ($\boldsymbol{\omega} \in \mathbb{R}^3$ para rotación y $\mathbf{u} \in \mathbb{R}^3$ para traslación) garantizando que la matriz resultante pertenezca a $\mathrm{SE}(3)$ de forma numéricamente estable (mediante expansiones de Taylor para ángulos $\theta \to 0$).
3. **Modos de Inicialización**:
   - `colmap`: Inicializa con las poses previas y aprende residuos $\exp(\boldsymbol{\xi}_i)$.
   - `identity`: Inicializa todas las cámaras en el origen del mundo ($\mathbf{I}_{4 \times 4}$), permitiendo reconstrucción y registro conjunto desde cero.
4. **Métricas de Evaluación de Trayectoria**: Integración de métricas de SLAM/NeRF:
   - **ATE** (*Absolute Trajectory Error*): RMSE de traslación y rotación tras alineación Sim(3) mediante el algoritmo de Umeyama.
   - **RPE** (*Relative Pose Error*): Error de odometría relativo entre fotogramas consecutivos.
5. **Soporte de Ground Truth (GT)**: Preservación del atributo `c2w_gt` a lo largo de los dataloaders (`reader_colmap_dataset.py`, `reader_nerf_dataset.py`) para monitorear la convergencia métrica en `test_stat/iterXXXXXX.json`.
6. **Interfaz CLI y Configuración YAML**: Activación mediante flags directos (`--pose_opt`, `--pose_init_mode`, etc.) o archivos de configuración (`cfg/pose_opt.yaml`).

---

## 2. Derivación Matemática y Arquitectura CUDA

### 2.1 Modelo de Rayo y Proyección
Para cada píxel en coordenadas de sensor normalizadas $\hat{\mathbf{v}} = (x, y, 1)^T$:
- Centro de la cámara (origen del rayo):
  $$\mathbf{r}_o = \mathbf{t}$$
- Dirección del rayo en coordenadas de mundo:
  $$\mathbf{r}_d = \mathbf{R} \, \hat{\mathbf{v}}$$
- Posición tridimensional del punto de muestreo $k$ a lo largo del rayo a distancia $s_k$:
  $$\mathbf{pt}_k = \mathbf{r}_o + s_k \mathbf{r}_d = \mathbf{t} + s_k (\mathbf{R} \, \hat{\mathbf{v}})$$

### 2.2 Gradiente Espacial de Densidad en Vóxeles
Dentro del vóxel, la coordenada local normalizada es $\mathbf{qt} \in [0, 1]^3$. La densidad $d$ se interpola trilinealmente a partir de los 8 vértices del vóxel. La función auxiliar `tri_interp_grad` en `cuda/src/auxiliary.h` evalúa:
$$\nabla_{\mathbf{pt}_k} d = \frac{1}{\text{vox\_l}} \nabla_{\mathbf{qt}_k} d$$
donde:
$$\frac{\partial d}{\partial x} = (1-y)(1-z)(p_1 - p_0) + y(1-z)(p_3 - p_2) + (1-y)z(p_5 - p_4) + yz(p_7 - p_6)$$
(y de forma análoga para $y$ y $z$).

### 2.3 Acumulación de Gradientes hacia el Rayo
Durante la integración de volumen en el backward pass, cada muestra contribuye al gradiente espacial:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{pt}_k} = \mathbf{g}_{pt, k}$$
Por la regla de la cadena:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{r}_o} = \sum_{k} \mathbf{g}_{pt, k}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{r}_d} = \sum_{k} s_k \mathbf{g}_{pt, k}$$

### 2.4 Gradiente respecto a los Extrínsecos $(\mathbf{R}, \mathbf{t})$
Dado que $\mathbf{r}_d = \mathbf{R} \hat{\mathbf{v}}$ y $\mathbf{r}_o = \mathbf{t}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{R}} = \frac{\partial \mathcal{L}}{\partial \mathbf{r}_d} \cdot \hat{\mathbf{v}}^T \quad \in \mathbb{R}^{3 \times 3}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{t}} = \frac{\partial \mathcal{L}}{\partial \mathbf{r}_o} \quad \in \mathbb{R}^{3 \times 1}$$

En `cuda/src/backward.cu`:
1. Cada hilo calcula su contribución de píxel al tensor $3 \times 4$:
   $$\mathbf{G}_{c2w}[i, j] = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{r}_d} \right)_i \cdot \hat{\mathbf{v}}_j, \quad \mathbf{G}_{c2w}[i, 3] = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{r}_o} \right)_i$$
2. Se realiza una reducción en memoria compartida por bloque:
   ```cuda
   __shared__ float s_block_c2w[12];
   ```
3. El primer hilo de cada bloque suma atómicamente a la matriz global en GPU:
   ```cuda
   atomicAdd(&dL_dc2w[entry], s_block_c2w[entry]);
   ```

### 2.5 Aislamiento de Armónicos Esféricos (SH Detach)
Para evitar que la dependencia direccional de los armónicos esféricos ($c2w[:3, 3]$ en el cálculo de dirección de vista) genere gradientes ruidosos de alta frecuencia en la traslación, `cam_pos` se desacopla explícitamente (`.detach()`) al evaluar `vox_fn` y `SH_eval`, obligando a que la señal geométrica guíe la pose.

---

## 3. Inventario de Archivos y Modificaciones

### 3.1 Kernel CUDA y Bindings C++
- [`cuda/src/auxiliary.h`](file:///home/felix/Projects/svraster/cuda/src/auxiliary.h):
  - Añadida la función inline `tri_interp_grad(const float3 qt, const float geo_params[8])` para obtener las derivadas analíticas $(\frac{\partial d}{\partial x}, \frac{\partial d}{\partial y}, \frac{\partial d}{\partial z})$ del campo de densidad.
- [`cuda/src/backward.cu`](file:///home/felix/Projects/svraster/cuda/src/backward.cu):
  - Añadido puntero `float* dL_dc2w` a los kernels `renderCUDA` y a la función `render`.
  - Inicialización de acumuladores por rayo `dL_dro` y `dL_drd`.
  - Acumulación en cuadratura por muestra:
    ```cuda
    dL_dro.x += dL_dpt.x; dL_dro.y += dL_dpt.y; dL_dro.z += dL_dpt.z;
    dL_drd.x += s * dL_dpt.x; dL_drd.y += s * dL_dpt.y; dL_drd.z += s * dL_dpt.z;
    ```
  - Ámbito de gradientes de profundidad: `float dLdepth_dI[3] = {0.f, 0.f, 0.f};` declarado en el ámbito exterior de cada vóxel para permitir la fusión de derivadas de profundidad en `dL_dpt` tanto si `need_depth` está activo como si no, evitando errores de identificador no definido en `nvcc`.
  - Reducción en bloque y suma atómica a `dL_dc2w` ($3 \times 4$).
  - `rasterize_voxels_backward` ahora crea el tensor de salida:
    ```cpp
    torch::Tensor dL_dc2w = torch::zeros({3, 4}, c2w_matrix.options());
    ```
    y retorna la tupla de 4 tensores: `(dL_dgeos, dL_drgbs, subdiv_p_bw, dL_dc2w)`.
- [`cuda/src/backward.h`](file:///home/felix/Projects/svraster/cuda/src/backward.h):
  - Declaración actualizada de `rasterize_voxels_backward` retornando `std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>`.
- [`cuda/svraster_cuda/renderer.py`](file:///home/felix/Projects/svraster/cuda/svraster_cuda/renderer.py):
  - Inclusión de `raster_settings.c2w_matrix` en `_RasterizeVoxels.apply(...)`.
  - Desempaquetado dinámico y seguro en `_RasterizeVoxels.backward(...)`: verifica si `_C.rasterize_voxels_backward` retorna 4 tensores (extensión actualizada) o 3 tensores (binario previo sin recompilar), asignando `dL_dc2w = None` y emitiendo advertencia instructiva en caso de no recompilación para evitar excepciones de tipo `ValueError`.
  - Retorno de `dL_dc2w` en la posición 9 correspondiente al tensor `c2w_matrix`.
  - `cam_pos = raster_settings.c2w_matrix[:3, 3].detach()` para aislar la rama de armónicos esféricos.

### 3.2 Utilidades de Álgebra de Lie y Métricas
- [`src/utils/camera_utils.py`](file:///home/felix/Projects/svraster/src/utils/camera_utils.py):
  - `skew_symmetric(w)`: Construye la matriz antisimétrica $[\mathbf{w}]_\times$.
  - `so3_to_SO3(w)`: Mapa exponencial de Rodrigues con serie de Taylor de 2do orden para $\theta < 10^{-5}$.
  - `se3_to_SE3(wu)`: Mapa exponencial de $\mathfrak{se}(3) \to \mathrm{SE}(3)$ con matriz de integración $\mathbf{V}$. Diferenciable por autograd.
  - `umeyama_alignment(X, Y, with_scale=True)`: Estimación cerrada de escala, rotación y traslación $\mathrm{Sim}(3)$ por descomposición SVD.
  - `compute_ate(pred_c2ws, gt_c2ws, align=True)`: Retorna `{'trans_rmse': float, 'rot_rmse_deg': float}`.
  - `compute_rpe(pred_c2ws, gt_c2ws)`: Retorna el error relativo entre fotogramas consecutivos.

### 3.3 Módulos de Cámara y Datasets
- [`src/cameras.py`](file:///home/felix/Projects/svraster/src/cameras.py):
  - Soporte de `c2w_gt` en la clase `Camera`.
  - Implementación de la clase [`CameraPoseOptimizer`](file:///home/felix/Projects/svraster/src/cameras.py#L293):
    - Parámetro entrenable `self.se3_refine = torch.nn.Embedding(num_cams, 6)`.
    - Buffer `self.base_c2w`: Poses fijas base (provenientes de COLMAP o $\mathbf{I}_{4 \times 4}$ en modo identity).
    - Método `get_c2w(cam_idx)`: Evalúa $\mathbf{T}_{base} \cdot \exp(\boldsymbol{\xi})$.
    - Métodos `get_all_c2w()`, `state_dict_poses()`, `load_state_dict_poses()`.
- [`src/dataloader/data_pack.py`](file:///home/felix/Projects/svraster/src/dataloader/data_pack.py):
  - Traspaso del atributo `c2w_gt` en `CameraCreator` y al instanciar `Camera`.
- [`src/dataloader/reader_colmap_dataset.py`](file:///home/felix/Projects/svraster/src/dataloader/reader_colmap_dataset.py) y [`reader_nerf_dataset.py`](file:///home/felix/Projects/svraster/src/dataloader/reader_nerf_dataset.py):
  - Lectura y almacenamiento de poses de referencia ground truth cuando existan.

### 3.4 Configuración y Entrenamiento
- [`src/config.py`](file:///home/felix/Projects/svraster/src/config.py):
  - Nueva sección `_C.pose_opt`:
    - `pose_opt` (bool, default `False`)
    - `pose_init_mode` (str, default `'colmap'`)
    - `lr_pose` (float, default `1e-3`)
    - `lr_pose_end` (float, default `1e-5`)
    - `warmup_pose` (int, default `500`)
    - `eval_pose_gt` (bool, default `True`)
- [`cfg/pose_opt.yaml`](file:///home/felix/Projects/svraster/cfg/pose_opt.yaml):
  - Archivo de configuración YAML listo para usar.
- [`train.py`](file:///home/felix/Projects/svraster/train.py):
  - Argumentos CLI añadidos al parser: `--pose_opt`, `--pose_init_mode`, `--lr_pose`, `--warmup_pose`.
  - Instanciación de `CameraPoseOptimizer`, optimizador `Adam` y scheduler exponencial (`sched_pose`).
  - Actualización dinámica por iteración:
    ```python
    if cfg.pose_opt.pose_opt and pose_optimizer is not None:
        cam_c2w = pose_optimizer.get_c2w(cam_idx)
        cam.c2w = cam_c2w
        cam.w2c = torch.inverse(cam_c2w).detach()
    ```
  - Zero-grad conjunto y backward pass en PyTorch.
  - Step condicional al periodo de warmup (`iteration > cfg.pose_opt.warmup_pose`).
  - Guardado y restauración de checkpoints de pose (`pose_opt_XXXXXX.pt` y `pose_opt.pt`).
  - Evaluación métrica de ATE y RPE en `training_report` volcada a `test_stat/iterXXXXXX.json`.

---

## 4. Compilación del Módulo CUDA

Para compilar el módulo de extensión `svraster_cuda` con los kernels actualizados, ejecute en el directorio del proyecto con el entorno `svraster`:

```bash
cd cuda
conda run -p /home/felix/miniconda3/envs/svraster python setup.py build_ext --inplace
```

*(Nota: Asegúrese de tener configurado `CUDA_HOME` apuntando al toolkit CUDA correspondiente si no se encuentra en las rutas estándar del sistema).*

---

## 5. Instrucciones de Uso

### 5.1 Modo 1: Activación mediante Archivo de Configuración YAML
```bash
conda run -p /home/felix/miniconda3/envs/svraster python train.py \
    --cfg_files cfg/pose_opt.yaml \
    --source_path <DATASET_PATH> \
    --model_path ./output/svraster_pose_opt
```

### 5.2 Modo 2: Activación Directa por Línea de Comandos (Refinamiento desde COLMAP)
```bash
conda run -p /home/felix/miniconda3/envs/svraster python train.py \
    --pose_opt \
    --pose_init_mode colmap \
    --lr_pose 0.001 \
    --warmup_pose 500 \
    --source_path <DATASET_PATH> \
    --model_path ./output/svraster_colmap_refine
```

### 5.3 Modo 3: Optimización desde Cero (Identity Initialization)
```bash
conda run -p /home/felix/miniconda3/envs/svraster python train.py \
    --pose_opt \
    --pose_init_mode identity \
    --lr_pose 0.002 \
    --warmup_pose 1000 \
    --source_path <DATASET_PATH> \
    --model_path ./output/svraster_from_scratch
```

### 5.4 Reanudación desde Checkpoint
```bash
conda run -p /home/felix/miniconda3/envs/svraster python train.py \
    --pose_opt \
    --load_iteration 15000 \
    --load_optimizer \
    --model_path ./output/svraster_pose_opt
```

---

## 6. Verificación y Resultados de Pruebas

Se ejecutaron pruebas automatizadas dentro del entorno `/home/felix/miniconda3/envs/svraster`:
1. **Propiedades de $\mathrm{SO}(3)$ y $\mathrm{SE}(3)$**:
   - Identidad: $\exp(\mathbf{0}) = \mathbf{I}_{4 \times 4}$.
   - Ortogonalidad: $\mathbf{R}^T \mathbf{R} = \mathbf{I}_3$, $\det(\mathbf{R}) = 1$ (error $< 10^{-6}$).
   - Estabilidad numérica en ángulos pequeños ($\theta < 10^{-5}$) validada mediante Taylor.
2. **Retropropagación de Autograd**:
   - Flujo de gradientes verificado desde la pérdida $\mathcal{L}(c2w)$ hasta `se3_refine.weight.grad`.
   - Aislamiento selectivo verificado: únicamente la cámara consultada en la iteración recibe gradientes; las demás permanecen en cero.
3. **Métricas de Evaluación**:
   - ATE y RPE evaluados con trayectorias sintéticas idénticas arrojaron error nulo ($0.0 \text{ m}, 0.0^\circ$).
   - Umeyama Sim(3) probado con rotaciones y traslaciones arbitrarias, recuperando la trayectoria de referencia con exactitud.

---

## 7. Registro de Métricas y Variación de Pose por Época

Para supervisar la convergencia simultánea de la estructura geométrica y de la estimación de las poses sin sobrecargar el almacenamiento en disco, se implementó un sistema de registro ligero y exhaustivo en `train.py` (`record_epoch_metrics`).

### 7.1 Filosofía de Diseño: Variación Promedio vs Archivos por Época
En lugar de volcar los extrínsecos completos ($[N, 4, 4]$) a archivos `.pt` en cada época (lo cual saturaría el disco con cientos de checkpoints redundantes), se registra la **variación promedio por época** y las **métricas esenciales de estructura y pose** en archivos tabulares y estructurados:
1. `metrics_per_epoch.csv`: Archivo CSV estándar, compatible directamente con `pandas.read_csv`, Excel o scripts de graficado con `matplotlib`.
2. `metrics_per_epoch.jsonl`: Archivo JSON Lines, ideal para procesamiento por lotes o dashboards dinámicos.
3. **Consola interactiva (`tqdm.write`)**: Notificación visual al finalizar cada época con resumen conciso sin interferir con la barra de progreso.

### 7.2 Definición de Época en SVRaster
En SVRaster, una época completa corresponde a un ciclo donde todas las cámaras de entrenamiento han sido muestreadas:
$$\text{Época } k \iff \text{Iteración } = k \times |\mathcal{C}_{\text{train}}|$$
(e.g., para un conjunto de 132 cámaras, la época 1 culmina en la iteración 132, la época 2 en la 264, etc.). También se registra obligatoriamente la última iteración del entrenamiento (`cfg.procedure.n_iter`).

### 7.3 Métricas Registradas
| Categoría | Campo | Descripción |
| :--- | :--- | :--- |
| **Tiempo / Época** | `epoch` | Número secuencial de época ($1, 2, \dots$) |
| | `iteration` | Iteración global de entrenamiento |
| | `elapsed_sec` | Tiempo acumulado de entrenamiento (segundos) |
| | `epoch_time_sec` | Tiempo transcurrido durante la época actual (segundos) |
| **Estructura** | `loss` | Pérdida combinada (suavizada por EMA) |
| | `psnr` | PSNR en decibelios (suavizado por EMA) |
| | `num_voxels` | Número total de vóxeles activos en el octree |
| | `inside_voxels` | Número de vóxeles en la región foreground (bounding box) |
| | `inside_pct` | Porcentaje de vóxeles en foreground ($\%$) |
| | `lr_geo` | Learning rate actual para los vértices de la cuadrícula de vóxeles |
| **Variación por Época** | `epoch_drot_mean_deg` | **Variación promedio de rotación** en la época actual ($\Delta \mathbf{R}$ respecto a la época anterior, grados) |
| | `epoch_drot_max_deg` | Variación máxima de rotación en la época actual (grados) |
| | `epoch_dtrans_mean_m` | **Variación promedio de traslación** en la época actual ($\Delta \mathbf{t}$ respecto a la época anterior, metros) |
| | `epoch_dtrans_max_m` | Variación máxima de traslación en la época actual (metros) |
| **Variación Acumulada** | `total_drot_mean_deg` | Rotación promedio acumulada respecto a la inicialización COLMAP (grados) |
| | `total_drot_max_deg` | Rotación máxima acumulada respecto a la inicialización (grados) |
| | `total_dtrans_mean_m` | Traslación promedio acumulada respecto a la inicialización (metros) |
| | `total_dtrans_max_m` | Traslación máxima acumulada respecto a la inicialización (metros) |
| | `lr_pose` | Learning rate actual del optimizador de pose ($0.0$ durante warmup) |
| **Trayectoria vs GT** | `ate_trans_rmse_m` | Absolute Trajectory Error de traslación con alineación Sim(3) (metros) |
| | `ate_rot_rmse_deg` | Absolute Trajectory Error de rotación con alineación Sim(3) (grados) |
| | `rpe_trans_rmse_m` | Relative Pose Error de traslación entre fotogramas consecutivos (metros) |
| | `rpe_rot_rmse_deg` | Relative Pose Error de rotación entre fotogramas consecutivos (grados) |

