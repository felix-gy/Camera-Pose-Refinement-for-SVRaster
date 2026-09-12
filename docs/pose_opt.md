# Plan de Implementación: Optimización de Poses de Cámara en SVRaster

Este documento detalla el análisis técnico, las decisiones arquitectónicas y el plan paso a paso para hacer que las **poses de cámara sean optimizables conjuntamente con el campo de vóxeles dispersos** durante el entrenamiento de **SVRaster**.

---

## 1. Diagnóstico Arquitectónico y Análisis Crítico del Código Propuesto

En métodos tipo NeRF basados en MLPs (como BARF, SC-NeRF o NeRF--), el trazado de rayos ocurre en tensores directos de PyTorch, permitiendo que `loss.backward()` propague gradientes a las matrices de cámara de forma automática. 

En **SVRaster**, el renderizado volumétrico ocurre íntegramente en un **Kernel C++/CUDA compilado** ([cuda/src/backward.cu](file:///home/felix/Projects/svraster/cuda/src/backward.cu)). Por esta razón, el código estándar de BARF **no funcionará de inmediato** sin habilitar el flujo de gradientes a nivel de CUDA y PyTorch Autograd.

### Análisis pieza por pieza de tu propuesta

```
┌────────────────────────────────────────────────────────┬─────────────┬────────────────────────────────────────────────────────┐
│ Componente Propuesto                                   │ Estado      │ Diagnóstico y Justificación                            │
├────────────────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ to_hom(X) y cam2world(X, pose)                         │ SOBRA       │ Innecesario. En SVRaster no transformamos puntos       │
│                                                        │             │ punto a punto en Python; el rasterizador recibe        │
│                                                        │             │ matrices 4x4 completas (c2w y w2c).                    │
├────────────────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ class Pose (invert, compose, etc.)                     │ SOBRA       │ Incompatible y redundante. Asume matrices [3, 4],      │
│                                                        │             │ mientras que SVRaster usa matrices homogéneas [4, 4].  │
│                                                        │             │ Las operaciones se hacen directamente con tensores:    │
│                                                        │             │ c2w_opt = c2w_base @ delta_T; torch.inverse().        │
├────────────────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ class Lie (Álgebra de Lie)                             │ PARCIAL     │ La formulación se(3) -> SE(3) es indispensable.        │
│                                                        │             │ Sobran SO3_to_so3 y SE3_to_se3 si siempre partimos     │
│                                                        │             │ del vector cero xi = 0.                                │
├────────────────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ torch.nn.Embedding(num_cams, 6)                        │ VÁLIDO      │ Estructura ideal. Mantiene un residuo se(3) de 6 DoF   │
│ inicializado en ceros                                  │             │ por cada cámara. Al iniciar en 0, exp(0) = I (identidad).│
├────────────────────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ setup_optimizer (Adam + ExponentialLR separado)        │ VÁLIDO      │ Indispensable. Las poses necesitan su propio Adam      │
│                                                        │             │ con learning rate bajo (1e-3 -> 1e-5) y scheduler exp. │
└────────────────────────────────────────────────────────┴─────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. El Reto en CUDA: ¿Dónde se corta el gradiente actualmente?

1. **Retorno en Autograd (`cuda/svraster_cuda/renderer.py:238`)**:
   En `_RasterizeVoxels.backward()`, el gradiente para `raster_settings` (donde viaja `c2w_matrix`) devuelve explícitamente `None`:
   ```python
   grads = (
       None, # => raster_settings (AQUÍ ESTÁ c2w_matrix)
       None, # => geomBuffer
       ...
       dL_dgeos,
       dL_drgbs,
       subdiv_p_bw,
   )
   ```
2. **Bloqueo en Armónicos Esféricos (`cuda/svraster_cuda/renderer.py:266`)**:
   En `SH_eval.forward()`, existe una guarda:
   ```python
   if torch.is_tensor(cam_pos) and cam_pos.requires_grad:
       raise NotImplementedError
   ```
3. **Ausencia de $\frac{\partial \mathcal{L}}{\partial c2w}$ en el Kernel (`cuda/src/backward.cu`)**:
   El kernel `BACKWARD::renderCUDA` lee `c2w_matrix` para proyectar el origen `ro` y la dirección `rd`, pero no acumula derivadas respecto a la pose.

---

## 3. Formulación Matemática Rigurosa del Backward en CUDA

Para cada píxel $p = (x, y)$ con rayo en espacio de cámara $\hat{v} = \frac{v_{cam}}{\|v_{cam}\|}$:
$$r_o = t = c2w_{[:3, 3]}$$
$$r_d = R \cdot \hat{v} = c2w_{[:3, :3]} \cdot \hat{v}$$

Cada muestra $k$ a lo largo del rayo dentro del vóxel tiene posición 3D:
$$pt_k = r_o + s_k \cdot r_d$$

### Derivada espacial de la densidad trilineal
Dentro del vóxel de tamaño $vox\_l$, las coordenadas normalizadas son:
$$qt_k = \frac{pt_k - (vox\_c - 0.5 \cdot vox\_l)}{vox\_l}$$
La densidad interpolada es $d(qt) = \sum_{m=0}^7 geo\_params[m] \cdot w_m(qt)$.  
Su gradiente analítico respecto a $qt$ es:
$$\nabla_{qt} d = \begin{pmatrix} 
\frac{\partial d}{\partial x} \\
\frac{\partial d}{\partial y} \\
\frac{\partial d}{\partial z}
\end{pmatrix}$$

Y la derivada respecto al punto en el mundo $pt_k$:
$$\nabla_{pt} d = \frac{1}{vox\_l} \nabla_{qt} d$$

### Derivadas de la pérdida respecto a $r_o$ y $r_d$
A través del volumen integrado $I$, para cada muestra $k$:
$$\frac{\partial \mathcal{L}}{\partial pt_k} = \left( \frac{\partial \mathcal{L}}{\partial I} + \frac{\partial \mathcal{L}_{depth}}{\partial I_k} \right) \cdot \frac{\partial I}{\partial d_k} \cdot \nabla_{pt} d_k$$
Dado que $pt_k = r_o + s_k \cdot r_d$:
$$\mathbf{g}_{ro} = \sum_{j, k} \frac{\partial \mathcal{L}}{\partial pt_{j, k}}$$
$$\mathbf{g}_{rd} = \sum_{j, k} s_{j, k} \cdot \frac{\partial \mathcal{L}}{\partial pt_{j, k}}$$

### Derivada respecto a la matriz $c2w \in \mathbb{R}^{3 \times 4}$
Aplicando la regla de la cadena respecto a $R$ y $t$:
$$\frac{\partial \mathcal{L}}{\partial t} = \sum_{\text{píxeles}} \mathbf{g}_{ro} \in \mathbb{R}^3$$
$$\frac{\partial \mathcal{L}}{\partial R} = \sum_{\text{píxeles}} \mathbf{g}_{rd} \cdot \hat{v}^T \in \mathbb{R}^{3 \times 3} \quad \left( \text{i.e. } \frac{\partial \mathcal{L}}{\partial R_{ij}} = (\mathbf{g}_{rd})_i \cdot \hat{v}_j \right)$$

Esta matriz de gradientes acumulada $\frac{\partial \mathcal{L}}{\partial c2w} \in \mathbb{R}^{4 \times 4}$ es retornada por el kernel CUDA hacia PyTorch.  
Dado que $c2w = c2w_{base} \cdot \exp(\hat{\xi})$, **PyTorch Autograd propaga automáticamente de $\frac{\partial \mathcal{L}}{\partial c2w}$ hacia los parámetros $\xi \in \mathbb{R}^6$**.

---

## 4. Estrategias de Inicialización y Soporte para Ground Truth (GT)

### A. Inicialización
1. **Modo `colmap` (Refinamiento)**:
   * $c2w_{base} = c2w_{colmap}$.
   * $\xi = \mathbf{0}_{6}$.
   * Ideal para corregir desviaciones milimétricas y de rotación en SfM ruidoso.
2. **Modo `identity` o `from_scratch` (Sin COLMAP)**:
   * $c2w_{base} = I_{4 \times 4}$ (o distribución esférica sintética mirando al origen).
   * $\xi = \mathbf{0}_{6}$.
   * **Requisito para no colapsar**: Activar priors monoculares como `--lambda_depthanythingv2` o `--lambda_mast3r_metric_depth`, y aplicar un período de calentamiento (`warmup_iter`) de 1500 iteraciones donde las poses permanezcan congeladas para consolidar la geometría base de los vóxeles.

### B. Soporte para Ground Truth (GT) y Métricas
Si el dataset incluye poses reales (por ejemplo de NeRF Synthetic, ScanNet++, 7-Scenes o blender):
* Se almacenan las poses $c2w_{gt}$ en el objeto `Camera`.
* Se calculan periódicamente las métricas estándar:
  * **ATE (Absolute Trajectory Error)**: Error medio cuadrático de traslación ($RMSE_{trans}$) tras alineación de Similitud $Sim(3)$ (Umeyama).
  * **RPE (Relative Pose Error)**: Error relativo de rotación ($^\circ$) y traslación ($cm$) entre pares de vistas consecutivas.

---

## 5. Plan de Modificaciones Detallado por Archivo

### Paso 1: Extensiones en CUDA (`cuda/src/`)
* [cuda/src/auxiliary.h](file:///home/felix/Projects/svraster/cuda/src/auxiliary.h):
  * Implementar `tri_interp_grad(qt, geo_params)` para calcular analíticamente $\nabla_{qt} d$.
* [cuda/src/backward.cu](file:///home/felix/Projects/svraster/cuda/src/backward.cu):
  * Agregar acumuladores `dL_dro` y `dL_drd` por hilo de píxel en `renderCUDA`.
  * Computar el producto exterior `pix_c2w_grad` ($3 \times 4$).
  * Reducción en memoria compartida por bloque y acumulación atómica (`atomicAdd`) en el buffer de salida `dL_dc2w`.
* [cuda/src/backward.h](file:///home/felix/Projects/svraster/cuda/src/backward.h) y [cuda/binding.cpp](file:///home/felix/Projects/svraster/cuda/binding.cpp):
  * Actualizar la signatura de `rasterize_voxels_backward` para recibir y retornar `dL_dc2w`.

### Paso 2: Capa PyTorch Autograd (`cuda/svraster_cuda/renderer.py`)
* En `_RasterizeVoxels`:
  * Pasar `raster_settings.c2w_matrix` como tensor directo diferenciable en `.apply()`.
  * En `.backward()`, capturar `dL_dc2w` de CUDA y retornarlo en el slot correspondiente a `c2w_matrix`.
* En `SH_eval`:
  * Pasar `cam_pos.detach()` para desacoplar el brillo especular de la optimización de posición y evitar inestabilidad.

### Paso 3: Álgebra de Lie y Módulo de Poses (`src/utils/` y `src/cameras.py`)
* Crear módulo de Lie en [src/utils/camera_utils.py](file:///home/felix/Projects/svraster/src/utils/camera_utils.py):
  * `skew_symmetric(w)`
  * `se3_to_SE3(wu)` (con Taylor para $\theta \to 0$).
  * Métricas de evaluación de trayectoria: `compute_ate(pred_c2w, gt_c2w)`.
* Implementar `CameraPoseOptimizer` en [src/cameras.py](file:///home/felix/Projects/svraster/src/cameras.py):
  * Embedding de $N \times 6$ con ceros.
  * Soporte de modos `colmap` vs `identity`.
  * Registro de tensores $c2w_{gt}$ para evaluación.

### Paso 4: Configuración del Sistema (`src/config.py` y `cfg/`)
* Agregar nodo `cfg.pose_opt` en [src/config.py](file:///home/felix/Projects/svraster/src/config.py):
  ```python
  cfg.pose_opt = CfgNode(dict(
      enable = False,
      init_mode = "colmap",        # "colmap" | "identity"
      lr_pose = 1e-3,
      lr_pose_end = 1e-5,
      warmup_iter = 1000,
      eval_gt = True,
  ))
  ```

### Paso 5: Integración en el Entrenamiento (`train.py`)
* Instanciar `CameraPoseOptimizer`, `torch.optim.Adam` y `ExponentialLR`.
* En cada iteración:
  * Si `iteration > warmup_iter`: actualizar la pose de la cámara actual con `c2w_opt = pose_optimizer.get_c2w(cam_idx)`.
  * Ejecutar `loss.backward()`, que ahora propaga gradientes tanto a los vóxeles como a `se3_refine.weight`.
  * Ejecutar `optimizer.step()` (vóxeles) y `optim_pose.step()` (cámara).
  * Reportar el error ATE/RPE periódicamente si las poses GT están disponibles.

---

## 6. Comandos CLI y Modo de Uso

Se activará la optimización de pose mediante argumentos de línea de comandos o archivos YAML:

```bash
# Caso 1: Refinamiento sobre COLMAP
python train.py --source_path $DATA_PATH --model_path $OUTPUT_PATH \
    --pose_opt.enable True \
    --pose_opt.init_mode colmap \
    --pose_opt.lr_pose 0.001 \
    --pose_opt.warmup_iter 1000

# Caso 2: Desde Cero (Sin poses de COLMAP, con priors monoculares)
python train.py --source_path $DATA_PATH --model_path $OUTPUT_PATH \
    --pose_opt.enable True \
    --pose_opt.init_mode identity \
    --pose_opt.lr_pose 0.0005 \
    --pose_opt.warmup_iter 1500 \
    --lambda_depthanythingv2 0.05
```
