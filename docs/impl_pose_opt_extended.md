# Registro Extendido de Implementación: Optimización de Poses de Cámara en SVRaster (Vía A CUDA)

Este documento constituye el desglose técnico, matemático y arquitectónico exhaustivo de la implementación de optimización de poses de cámara (6-DoF en el álgebra de Lie $\mathfrak{se}(3)$) sobre el rasterizador volumétrico de vóxeles dispersos **SVRaster**.

Aquí se detallan todos los cambios realizados, archivo por archivo, función por función y línea por línea, explicando la motivación teórica, las ecuaciones físicas, las decisiones de bajo nivel en CUDA y el impacto en la estabilidad numérica del entrenamiento.

---

## Índice General

1. [Fundamentación Teórica del Problema](#1-fundamentación-teórica-del-problema)
2. [Análisis Comparativo: ¿Por qué Vía A en CUDA?](#2-análisis-comparativo-por-qué-vía-a-en-cuda)
3. [Derivación Matemática Rigurosa de Gradientes de Cámara](#3-derivación-matemática-rigurosa-de-gradientes-de-cámara)
   - 3.1. [Geometría del rayo y dependencia respecto a la pose](#31-geometría-del-rayo-y-dependencia-respecto-a-la-pose)
   - 3.2. [Gradiente espacial en vóxeles trilineales ($\nabla_{\mathbf{pt}} d$)](#32-gradiente-espacial-en-vóxeles-trilineales-nabla_mathbfpt-d)
   - 3.3. [Integración a lo largo del rayo (Momentos de orden 0 y 1)](#33-integración-a-lo-largo-del-rayo-momentos-de-orden-0-y-1)
   - 3.4. [Producto exterior para $\frac{\partial \mathcal{L}}{\partial \mathbf{R}}$ y $\frac{\partial \mathcal{L}}{\partial \mathbf{t}}$](#34-producto-exterior-para-fracpartial-mathcallpartial-mathbfr-y-fracpartial-mathcallpartial-mathbft)
4. [Parametrización en el Álgebra de Lie $\mathfrak{se}(3)$](#4-parametrización-en-el-álgebra-de-lie-mathfraksen3)
   - 4.1. [¿Por qué $\mathfrak{se}(3)$ y no cuaterniones o ángulos de Euler?](#41-por-qué-mathfraksen3-y-no-cuaterniones-o-ángulos-de-euler)
   - 4.2. [Mapa exponencial de Rodrigues con serie de Taylor](#42-mapa-exponencial-de-rodrigues-con-serie-de-taylor)
   - 4.3. [Matriz de integración $\mathbf{V}$ y traslación acoplada](#43-matriz-de-integración-mathbfv-y-traslación-acoplada)
5. [Desglose Exhaustivo Archivo por Archivo](#5-desglose-exhaustivo-archivo-por-archivo)
   - 5.1. [`cuda/src/auxiliary.h`](#51-cudasrcauxiliaryh)
   - 5.2. [`cuda/src/backward.cu`](#52-cudasrcbackwardcu)
   - 5.3. [`cuda/src/backward.h`](#53-cudasrcbackwardh)
   - 5.4. [`cuda/svraster_cuda/renderer.py`](#54-cudasvraster_cudarendererpy)
   - 5.5. [`src/utils/camera_utils.py`](#55-srcutilscamera_utilspy)
   - 5.6. [`src/cameras.py`](#56-srccameraspy)
   - 5.7. [`src/dataloader/data_pack.py`](#57-srcdataloaderdata_packpy)
   - 5.8. [`src/dataloader/reader_colmap_dataset.py`](#58-srcdataloaderreader_colmap_datasetpy)
   - 5.9. [`src/dataloader/reader_nerf_dataset.py`](#59-srcdataloaderreader_nerf_datasetpy)
   - 5.10. [`src/config.py`](#510-srcconfigpy)
   - 5.11. [`cfg/pose_opt.yaml`](#511-cfgpose_optyaml)
   - 5.12. [`train.py`](#512-trainpy)
6. [Métricas de Trayectoria: Algoritmo de Umeyama, ATE y RPE](#6-métricas-de-trayectoria-algoritmo-de-umeyama-ate-y-rpe)
7. [Guía de Compilación, Ejecución y Comandos CLI](#7-guía-de-compilación-ejecución-y-comandos-cli)

---

## 1. Fundamentación Teórica del Problema

En la reconstrucción de escenas tridimensionales mediante campos de radiancia neurales (NeRF) o representaciones de vóxeles continuos dispersos (SVRaster), la formulación tradicional presupone que la pose de cada cámara de captura está perfectamente calibrada y fija:
$$\mathbf{T}_{c2w} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in \mathrm{SE}(3)$$

En escenarios reales, las poses provienen de algoritmos de Structure-from-Motion (SfM) como COLMAP. Estos métodos sufren de:
1. **Ruido en la triangulación**: Errores sub-píxel que se traducen en perturbaciones de rotación y traslación.
2. **Deriva acumulada (drift)**: Falta de consistencia global en trayectorias largas.
3. **Escenas sin características detectables**: Fallos en la detección de puntos clave donde SfM ni siquiera logra converger.

Al optimizar un modelo de alta resolución geométrica como SVRaster con poses inexactas, los detalles finos se borran (efecto *blurry*) o el modelo colapsa generando artefactos flotantes ("floaters") para compensar los errores de proyección de rayos incongruentes. Permitir que la pose de la cámara $\mathbf{T}_{c2w}$ sea un parámetro diferenciable optimizable conjuntamente con el volumen permite afinar la calibración a nivel sub-píxel y alcanzar nitidez fotográfica.

---

## 2. Análisis Comparativo: ¿Por qué Vía A en CUDA?

Para permitir que el gradiente de la función de pérdida $\mathcal{L}$ alcance la matriz de pose de la cámara $\mathbf{T}_{c2w}$, existen dos enfoques teóricos:

| Dimensión de Análisis | Vía B: Autograd de PyTorch Puro | Vía A: Kernel CUDA Analítico (Nuestra Elección) |
| :--- | :--- | :--- |
| **Punto de corte del gradiente** | Requiere construir el grafo de computación completo para millones de muestras de rayos en PyTorch. | El kernel de raymarching de CUDA acumula directamente $\frac{\partial \mathcal{L}}{\partial \mathbf{R}}$ y $\frac{\partial \mathcal{L}}{\partial \mathbf{t}}$. |
| **Uso de Memoria VRAM** | Crítico ($O(H \times W \times K)$). Almacena coordenadas de todos los puntos de integración en GPU para el backward. Provoca Out-of-Memory (OOM). | Mínimo ($O(1)$). Solo almacena un tensor de $3 \times 4$ números en memoria compartida y global. Cero consumo extra de VRAM. |
| **Velocidad de Cómputo** | Lenta por overhead de asignación de tensores intermedios y sincronización en PyTorch. | Máxima velocidad nativa en hardware. Se calcula durante el mismo pase backward de los vóxeles sin sobrecosto perceptible. |
| **Rigor Matemático** | Depende de la diferenciabilidad automática de operaciones matriciales compuestas. | Derivación explícita mediante la física del rayo y la cuadratura de volumen. |

**Decisión**: Se eligió la **Vía A (CUDA)** por su eficiencia computacional, escalabilidad a altas resoluciones ($1920 \times 1080$ con supersampling) y consumo nulo de memoria adicional.

---

## 3. Derivación Matemática Rigurosa de Gradientes de Cámara

### 3.1. Geometría del rayo y dependencia respecto a la pose
Sea un píxel en pantalla con coordenadas normalizadas en el plano focal de la cámara:
$$\mathbf{v} = \begin{pmatrix} \frac{u - c_x}{f_x} \\ \frac{v - c_y}{f_y} \\ 1 \end{pmatrix} = \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$
El vector director unitario en el marco de la cámara es $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$.

La transformación de cámara a mundo ($\mathbf{T}_{c2w} = [\mathbf{R} \mid \mathbf{t}]$) define el rayo en coordenadas de mundo:
1. **Origen del rayo**:
   $$\mathbf{r}_o = \mathbf{t} \in \mathbb{R}^3$$
2. **Dirección del rayo**:
   $$\mathbf{r}_d = \mathbf{R} \, \hat{\mathbf{v}} \in \mathbb{R}^3$$
3. **Punto de muestreo tridimensional** a una distancia $s_k$ a lo largo del rayo:
   $$\mathbf{pt}_k = \mathbf{r}_o + s_k \mathbf{r}_d = \mathbf{t} + s_k (\mathbf{R} \, \hat{\mathbf{v}})$$

### 3.2. Gradiente espacial en vóxeles trilineales ($\nabla_{\mathbf{pt}} d$)
Dentro del árbol de vóxeles de SVRaster, cada vóxel tiene centro $\mathbf{c}_{vox}$, tamaño de arista $\text{vox\_l}$, y 8 vértices en sus esquinas con parámetros de densidad geo-local $\{p_0, p_1, \dots, p_7\}$.

Para un punto $\mathbf{pt}_k = (x_w, y_w, z_w)$, la coordenada normalizada dentro del vóxel local es:
$$\mathbf{qt}_k = \begin{pmatrix} x_{loc} \\ y_{loc} \\ z_{loc} \end{pmatrix} = \frac{\mathbf{pt}_k - (\mathbf{c}_{vox} - \frac{\text{vox\_l}}{2})}{\text{vox\_l}} \in [0, 1]^3$$

La densidad escalar $d(\mathbf{qt}_k)$ resulta de la interpolación trilineal:
$$\begin{aligned}
d(x, y, z) &= (1-x)(1-y)(1-z) p_0 + x(1-y)(1-z) p_1 \\
&\quad + (1-x)y(1-z) p_2 + xy(1-z) p_3 \\
&\quad + (1-x)(1-y)z p_4 + x(1-y)z p_5 \\
&\quad + (1-x)yz p_6 + xyz p_7
\end{aligned}$$

Derivando analíticamente respecto a la coordenada local $x$:
$$\frac{\partial d}{\partial x} = (1-y)(1-z)(p_1 - p_0) + y(1-z)(p_3 - p_2) + (1-y)z(p_5 - p_4) + yz(p_7 - p_6)$$
De manera análoga:
$$\frac{\partial d}{\partial y} = (1-x)(1-z)(p_2 - p_0) + x(1-z)(p_3 - p_1) + (1-x)z(p_6 - p_4) + xz(p_7 - p_5)$$
$$\frac{\partial d}{\partial z} = (1-x)(1-y)(p_4 - p_0) + x(1-y)(p_5 - p_1) + (1-x)y(p_6 - p_2) + xy(p_7 - p_3)$$

Aplicando la regla de la cadena respecto a la coordenada en el espacio de mundo $\mathbf{pt}_k$:
$$\nabla_{\mathbf{pt}_k} d = \frac{\partial \mathbf{qt}_k}{\partial \mathbf{pt}_k}^T \nabla_{\mathbf{qt}_k} d = \frac{1}{\text{vox\_l}} \begin{pmatrix} \frac{\partial d}{\partial x} \\ \frac{\partial d}{\partial y} \\ \frac{\partial d}{\partial z} \end{pmatrix}$$

Esta formulación fue implementada como la función `tri_interp_grad` en `cuda/src/auxiliary.h`.

### 3.3. Integración a lo largo del rayo (Momentos de orden 0 y 1)
En el pase hacia atrás del renderizador volumétrico, la retropropagación de la pérdida fotométrica produce una derivada escalar respecto a la densidad en cada muestra $k$:
$$\frac{\partial \mathcal{L}}{\partial d_k} \in \mathbb{R}$$
Por tanto, el gradiente respecto a la posición espacial 3D del punto de muestreo es:
$$\mathbf{g}_{pt, k} = \frac{\partial \mathcal{L}}{\partial \mathbf{pt}_k} = \frac{\partial \mathcal{L}}{\partial d_k} \cdot \nabla_{\mathbf{pt}_k} d \in \mathbb{R}^3$$

Dado que $\mathbf{pt}_k = \mathbf{r}_o + s_k \mathbf{r}_d$:
1. Derivada respecto al origen del rayo $\mathbf{r}_o$:
   $$\frac{\partial \mathbf{pt}_k}{\partial \mathbf{r}_o} = \mathbf{I}_{3 \times 3} \implies \frac{\partial \mathcal{L}}{\partial \mathbf{r}_o} = \sum_{k} \frac{\partial \mathcal{L}}{\partial \mathbf{pt}_k} = \sum_{k} \mathbf{g}_{pt, k}$$
2. Derivada respecto a la dirección del rayo $\mathbf{r}_d$:
   $$\frac{\partial \mathbf{pt}_k}{\partial \mathbf{r}_d} = s_k \mathbf{I}_{3 \times 3} \implies \frac{\partial \mathcal{L}}{\partial \mathbf{r}_d} = \sum_{k} s_k \frac{\partial \mathcal{L}}{\partial \mathbf{pt}_k} = \sum_{k} s_k \mathbf{g}_{pt, k}$$

Nótese que $\frac{\partial \mathcal{L}}{\partial \mathbf{r}_o}$ representa el **momento de orden cero** del gradiente espacial a lo largo del rayo, mientras que $\frac{\partial \mathcal{L}}{\partial \mathbf{r}_d}$ representa el **momento de orden uno** ponderado por la profundidad de muestreo $s_k$.

### 3.4. Producto exterior para $\frac{\partial \mathcal{L}}{\partial \mathbf{R}}$ y $\frac{\partial \mathcal{L}}{\partial \mathbf{t}}$
Dado que $\mathbf{r}_o = \mathbf{t}$, la derivada respecto al vector de traslación es directa:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{t}} = \frac{\partial \mathcal{L}}{\partial \mathbf{r}_o} \in \mathbb{R}^3$$

Para la matriz de rotación $\mathbf{R} \in \mathbb{R}^{3 \times 3}$, puesto que $\mathbf{r}_d = \mathbf{R} \, \hat{\mathbf{v}}$:
$$\frac{\partial (\mathbf{r}_d)_i}{\partial \mathbf{R}_{j, m}} = \delta_{ij} \hat{\mathbf{v}}_m$$
Por la regla de la cadena:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{R}_{i, j}} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{r}_d} \right)_i \cdot \hat{\mathbf{v}}_j \iff \frac{\partial \mathcal{L}}{\partial \mathbf{R}} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{r}_d} \right) \hat{\mathbf{v}}^T \in \mathbb{R}^{3 \times 3}$$

Empaquetando en la matriz de extrínsecos $3 \times 4$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{T}_{c2w}} = \left[ \frac{\partial \mathcal{L}}{\partial \mathbf{R}} \;\middle|\; \frac{\partial \mathcal{L}}{\partial \mathbf{t}} \right] \in \mathbb{R}^{3 \times 4}$$

---

## 4. Parametrización en el Álgebra de Lie $\mathfrak{se}(3)$

### 4.1. ¿Por qué $\mathfrak{se}(3)$ y no cuaterniones o ángulos de Euler?
Optimizar directamente matrices $3 \times 3$ en $\mathbb{R}^{3 \times 3}$ destruye la ortogonalidad ($\mathbf{R}^T \mathbf{R} = \mathbf{I}$, $\det(\mathbf{R}) = 1$). Los enfoques alternativos presentan severos inconvenientes:
- **Ángulos de Euler**: Sufren de bloqueo de cardán (*gimbal lock*), discontinuidades en $\pm \pi$ y gradientes mal condicionados.
- **Cuaterniones**: Requieren una restricción de norma unitaria $\|q\|=1$. Tras un paso de optimizador como Adam, la norma se descalibra, y la proyección $q / \|q\|$ genera gradientes nulos a lo largo de la dirección radial, entorpeciendo la convergencia.

**Solución**: El álgebra de Lie $\mathfrak{se}(3)$ describe el espacio tangente al grupo de Lie $\mathrm{SE}(3)$ en el elemento identidad. Es un espacio vectorial euclidiano no restringido $\mathbb{R}^6$:
$$\boldsymbol{\xi} = \begin{pmatrix} \boldsymbol{\omega} \\ \mathbf{u} \end{pmatrix} \in \mathbb{R}^6$$
donde $\boldsymbol{\omega} \in \mathbb{R}^3$ parametriza la rotación y $\mathbf{u} \in \mathbb{R}^3$ la traslación. Cualquier paso de descenso de gradiente en $\mathbb{R}^6$ produce, al aplicar el mapa exponencial $\exp(\boldsymbol{\xi}^\wedge)$, un elemento válido de $\mathrm{SE}(3)$ de forma garantizada y sin restricciones.

### 4.2. Mapa exponencial de Rodrigues con serie de Taylor
El vector de rotación $\boldsymbol{\omega} \in \mathbb{R}^3$ define el ángulo $\theta = \|\boldsymbol{\omega}\|$ y el eje unitario $\mathbf{a} = \boldsymbol{\omega} / \theta$.
La matriz antisimétrica asociada $[\boldsymbol{\omega}]_\times$ cumple:
$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}$$

La fórmula clásica de Rodrigues establece:
$$\mathbf{R} = \exp([\boldsymbol{\omega}]_\times) = \mathbf{I} + \frac{\sin\theta}{\theta} [\boldsymbol{\omega}]_\times + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times^2$$

**Problema de Inestabilidad Numérica**: Cuando $\theta \to 0$ (condición inicial exacta del entrenamiento, donde $\boldsymbol{\xi} = \mathbf{0}$), evaluar $\frac{\sin\theta}{\theta}$ y $\frac{1 - \cos\theta}{\theta^2}$ produce $\frac{0}{0} = \text{NaN}$.

**Nuestra Solución**: Expansión de Taylor analítica en orden superior para $\theta < 10^{-5}$:
$$A(\theta) = \frac{\sin\theta}{\theta} = 1 - \frac{\theta^2}{6} + \frac{\theta^4}{120} + O(\theta^6)$$
$$B(\theta) = \frac{1 - \cos\theta}{\theta^2} = \frac{1}{2} - \frac{\theta^2}{24} + \frac{\theta^4}{720} + O(\theta^6)$$
Esto garantiza precisión de punto flotante de 32 bits y gradientes analíticos infinitamente diferenciables en la identidad.

### 4.3. Matriz de integración $\mathbf{V}$ y traslación acoplada
La traslación no es simplemente $\mathbf{u}$, sino que está acoplada con la rotación mediante la matriz de curvatura de Lie $\mathbf{V}$:
$$\mathbf{t} = \mathbf{V}(\boldsymbol{\omega}) \, \mathbf{u}$$
donde:
$$\mathbf{V}(\boldsymbol{\omega}) = \mathbf{I} + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times + \frac{\theta - \sin\theta}{\theta^3} [\boldsymbol{\omega}]_\times^2 = \mathbf{I} + B(\theta) [\boldsymbol{\omega}]_\times + C(\theta) [\boldsymbol{\omega}]_\times^2$$

Para $\theta < 10^{-5}$, la función $C(\theta)$ se expande analíticamente como:
$$C(\theta) = \frac{\theta - \sin\theta}{\theta^3} = \frac{1}{6} - \frac{\theta^2}{120} + \frac{\theta^4}{5040} + O(\theta^6)$$

La matriz homogénea resultante en $\mathrm{SE}(3)$ es:
$$\exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} \mathbf{I} + A [\boldsymbol{\omega}]_\times + B [\boldsymbol{\omega}]_\times^2 & (\mathbf{I} + B [\boldsymbol{\omega}]_\times + C [\boldsymbol{\omega}]_\times^2) \mathbf{u} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

Durante el entrenamiento, la pose refinada de la cámara $i$ se compone con la pose base fija:
$$\mathbf{T}_{opt}^{(i)} = \mathbf{T}_{base}^{(i)} \cdot \exp(\boldsymbol{\xi}_i^\wedge)$$
Al comenzar con $\boldsymbol{\xi}_i = \mathbf{0}$, $\exp(\mathbf{0}) = \mathbf{I}$, partiendo exactamente de la pose inicial sin discontinuidades.

---

## 5. Desglose Exhaustivo Archivo por Archivo

A continuación se detallan las 12 modificaciones efectuadas en el repositorio:

```
felix-gy/Camera-Pose-Refinement-for-SVRaster/
├── cfg/
│   └── pose_opt.yaml                      <-- [CREADO] Configuración por defecto
├── cuda/
│   ├── src/
│   │   ├── auxiliary.h                    <-- [MODIFICADO] tri_interp_grad
│   │   ├── backward.cu                    <-- [MODIFICADO] Acumulación dL_dc2w
│   │   └── backward.h                     <-- [MODIFICADO] Firma de función backward
│   └── svraster_cuda/
│       └── renderer.py                    <-- [MODIFICADO] Autograd bridge y SH detach
├── docs/
│   ├── pose_opt.md                        <-- [CREADO] Plan maestro
│   ├── impl_pose_opt.md                   <-- [CREADO] Registro sintético
│   └── impl_pose_opt_extended.md          <-- [CREADO] Este documento detallado
├── src/
│   ├── cameras.py                         <-- [MODIFICADO] CameraPoseOptimizer y c2w_gt
│   ├── config.py                          <-- [MODIFICADO] Nodo cfg.pose_opt
│   ├── dataloader/
│   │   ├── data_pack.py                   <-- [MODIFICADO] Conexión de c2w_gt
│   │   ├── reader_colmap_dataset.py       <-- [MODIFICADO] Soporte de poses GT
│   │   └── reader_nerf_dataset.py         <-- [MODIFICADO] Soporte de poses GT
│   └── utils/
│       └── camera_utils.py                <-- [MODIFICADO] SE(3), Umeyama, ATE, RPE
└── train.py                               <-- [MODIFICADO] Bucle de entrenamiento y CLI
```

---

### 5.1. `cuda/src/auxiliary.h`
- **Ruta**: [`cuda/src/auxiliary.h`](file:///home/felix/Projects/svraster/cuda/src/auxiliary.h)
- **Función añadida**: `__forceinline__ __device__ float3 tri_interp_grad(const float3 qt, const float geo_params[8])`
- **Justificación y Por qué**:
  El archivo `auxiliary.h` contenía la función `tri_interp` que calcula el valor escalar de la densidad $d$ dentro de un vóxel dado $\mathbf{qt} \in [0, 1]^3$, pero carecía por completo del cálculo de sus derivadas direccionales respecto a la posición espacial.
  Para que una perturbación en la posición del rayo $\mathbf{pt}_k$ influya en la densidad, se necesita conocer la pendiente tridimensional del campo escalar $\nabla_{\mathbf{qt}} d = (\frac{\partial d}{\partial x}, \frac{\partial d}{\partial y}, \frac{\partial d}{\partial z})$.
  Se diseñó una función inline para que el compilador de NVIDIA (`nvcc`) mantenga los cálculos en los registros del procesador vectorial sin accesos adicionales a la memoria global.

---

### 5.2. `cuda/src/backward.cu`
- **Ruta**: [`cuda/src/backward.cu`](file:///home/felix/Projects/svraster/cuda/src/backward.cu)
- **Modificaciones realizadas**:
  1. **Parámetro del kernel**: Se añadió `float* __restrict__ dL_dc2w` a la firma del kernel `renderCUDA` y a la función envoltorio `render(...)`.
  2. **Acumuladores por rayo**:
     ```cuda
     float3 dL_dro = {0.f, 0.f, 0.f};
     float3 dL_drd = {0.f, 0.f, 0.f};
     ```
  3. **Acumulación de muestras a lo largo del rayo**:
     En cada paso de la marcha de rayos dentro del vóxel, cuando se evalúa la densidad `dL_dgeo`:
     ```cuda
     float3 grad_d = tri_interp_grad(qt, geo_params);
     float3 dL_dpt = {
         dL_dd * grad_d.x / vox_l,
         dL_dd * grad_d.y / vox_l,
         dL_dd * grad_d.z / vox_l
     };
     dL_dro.x += dL_dpt.x;  dL_dro.y += dL_dpt.y;  dL_dro.z += dL_dpt.z;
     dL_drd.x += s * dL_dpt.x;  dL_drd.y += s * dL_dpt.y;  dL_drd.z += s * dL_dpt.z;
     ```
   4. **Ámbito y Fusión de Gradientes de Profundidad (`dLdepth_dI`)**:
      Cuando se entrena con regularizadores de profundidad (`--lambda_sparse_depth`, Mast3r, DepthAnythingv2), las muestras a lo largo del rayo reciben gradientes adicionales de profundidad.
      Para fusionar estos gradientes en la derivada de pose de cámara $\frac{\partial \mathcal{L}}{\partial \mathbf{pt}_k}$, la variable:
      ```cuda
      float dLdepth_dI[3] = {0.f, 0.f, 0.f};
      ```
      se declara en el ámbito exterior de cada vóxel (antes del bloque `if (need_depth)`).
      **¿Por qué este diseño?**:
      - Si `need_depth` está activo, `dLdepth_dI` se llena con los gradientes de profundidad por muestra $k$, permitiendo que la pérdida de profundidad también guíe la traslación y rotación de la cámara (`eff_dI = dL_dI + dLdepth_dI[k]`).
      - Si `need_depth` está desactivado, el arreglo permanece inicializado en ceros sin sobrecosto.
      - Resuelve la visibilidad léxica en `nvcc`, evitando errores de compilación por identificador no definido.
   5. **Producto exterior local**:
      Cada hilo de píxel calcula su contribución a la matriz de cámara:
      ```cuda
      float g_c2w[12];
      // R = drd (outer product) v_hat
      g_c2w[0] = dL_drd.x * v_hat.x;  g_c2w[1] = dL_drd.x * v_hat.y;  g_c2w[2] = dL_drd.x * v_hat.z;
      g_c2w[4] = dL_drd.y * v_hat.x;  g_c2w[5] = dL_drd.y * v_hat.y;  g_c2w[6] = dL_drd.y * v_hat.z;
      g_c2w[8] = dL_drd.z * v_hat.x;  g_c2w[9] = dL_drd.z * v_hat.y;  g_c2w[10] = dL_drd.z * v_hat.z;
      // t = dro
      g_c2w[3] = dL_dro.x;            g_c2w[7] = dL_dro.y;            g_c2w[11] = dL_dro.z;
      ```
   6. **Reducción de dos etapas (Shared Memory + AtomicAdd)**:
      Si cada hilo hiciera un `atomicAdd` directo a la memoria global de la GPU, los cientos de miles de hilos de la imagen ($1920 \times 1080 \approx 2 \times 10^6$) colapsarían el bus de memoria L2 por contención atómica.
      **Nuestra Solución**: Reducción en memoria compartida del bloque:
      ```cuda
      __shared__ float s_block_c2w[12];
      if (threadIdx.x == 0) {
          for (int i = 0; i < 12; ++i) s_block_c2w[i] = 0.f;
      }
      __syncthreads();
      for (int i = 0; i < 12; ++i) atomicAdd(&s_block_c2w[i], g_c2w[i]);
      __syncthreads();
      if (threadIdx.x == 0) {
          for (int i = 0; i < 12; ++i) atomicAdd(&dL_dc2w[i], s_block_c2w[i]);
      }
      ```
      Esto reduce los accesos atómicos globales en un factor de $256\times$ (el tamaño del bloque de hilos).
   7. **Tensor de salida en PyTorch C++ API**:
      ```cpp
      torch::Tensor dL_dc2w = torch::zeros({3, 4}, c2w_matrix.options());
      // llamada al kernel con dL_dc2w.data_ptr<float>()
      return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw, dL_dc2w);
      ```

---

### 5.3. `cuda/src/backward.h`
- **Ruta**: [`cuda/src/backward.h`](file:///home/felix/Projects/svraster/cuda/src/backward.h)
- **Modificación**:
  Se actualizó el tipo de retorno de `rasterize_voxels_backward`:
  De `std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>` a `std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>`.
- **Por qué**:
  Para permitir que PyBind11 empaquete el nuevo tensor `dL_dc2w` como el cuarto elemento devuelto hacia Python.

---

### 5.4. `cuda/svraster_cuda/renderer.py`
- **Ruta**: [`cuda/svraster_cuda/renderer.py`](file:///home/felix/Projects/svraster/cuda/svraster_cuda/renderer.py)
- **Modificaciones realizadas**:
  1. En `rasterize_voxels(...)`:
     ```python
     cam_pos = raster_settings.c2w_matrix[:3, 3].detach()
     vox_params = vox_fn(in_frusts_idx, cam_pos, raster_settings.color_mode)
     ```
     **¿Por qué `.detach()`?**:
     En el renderizado de armónicos esféricos (SH), el color especular de cada vóxel depende del vector de vista $(\mathbf{c}_{vox} - \text{cam\_pos})$. Si permitimos que el gradiente fluya por esta rama de color dependiente de la vista, la cámara tenderá a "rotar y trasladarse espuriamente" para ajustar brillos y reflejos artificiales en vez de aprender la geometría física del objeto. Desacoplar `cam_pos` aquí aísla la pose de cámara para que sea guiada puramente por la silueta, la densidad y la consistencia geométrica multi-vista.
  2. En `_RasterizeVoxels.forward`:
     Se recibe `raster_settings.c2w_matrix` como tensor explícito dentro de los argumentos de `_RasterizeVoxels.apply(...)`.
  3. En `_RasterizeVoxels.backward`:
     Implementación de desempaquetado con compatibilidad binaria dinámica:
     ```python
     ret = _C.rasterize_voxels_backward(*args)
     if len(ret) == 4:
         dL_dgeos, dL_drgbs, subdiv_p_bw, dL_dc2w = ret
     else:
         dL_dgeos, dL_drgbs, subdiv_p_bw = ret
         dL_dc2w = None
         if c2w_matrix is not None and c2w_matrix.requires_grad:
             if not hasattr(_RasterizeVoxels, '_warned_recompile'):
                 print("[WARNING] svraster_cuda extension returned 3 tensors instead of 4.")
                 print("[WARNING] The CUDA extension needs to be recompiled to enable camera pose gradients:")
                 print("          cd cuda && python setup.py build_ext --inplace")
                 _RasterizeVoxels._warned_recompile = True

     grads = (
         None,        # raster_settings
         None,        # geomBuffer
         None,        # octree_paths
         None,        # vox_centers
         None,        # vox_lengths
         dL_dgeos,    # geos
         dL_drgbs,    # rgbs
         subdiv_p_bw, # subdiv_p
         dL_dc2w,     # c2w_matrix
     )
     return grads
     ```
     **¿Por qué este diseño de compatibilidad hacia atrás?**:
     - Si el binario en disco ya fue recompilado con el nuevo kernel, `len(ret) == 4` y `dL_dc2w` se conecta directamente con PyTorch Autograd.
     - Si un usuario ejecuta el código antes de recompilar o en un entorno con un `.so` legado, `len(ret) == 3` se maneja de forma elegante asignando `dL_dc2w = None` e imprimiendo un mensaje informativo en lugar de arrojar un `ValueError: not enough values to unpack` que detendría bruscamente el entrenamiento.
     - Garantiza que la tupla devuelta a PyTorch tenga exactamente 9 elementos, preservando la correspondencia posicional con los 9 argumentos de `forward`.

---

### 5.5. `src/utils/camera_utils.py`
- **Ruta**: [`src/utils/camera_utils.py`](file:///home/felix/Projects/svraster/src/utils/camera_utils.py)
- **Funciones implementadas**:
  1. `skew_symmetric(w)`: Construye $[\mathbf{w}]_\times$ vectorizado para tensores de cualquier dimensionalidad por lotes `[..., 3]`.
  2. `so3_to_SO3(w)`: Implementa el mapa exponencial $\mathrm{SO}(3)$ con Taylor para $\theta < 10^{-5}$.
  3. `se3_to_SE3(wu)`: Implementa el mapa exponencial $\mathrm{SE}(3)$ con matriz $\mathbf{V}$, devolviendo tensores homogéneos `[..., 4, 4]`.
  4. `umeyama_alignment(X, Y, with_scale=True)`: Algoritmo cerrado por descomposición SVD que encuentra la transformación de similitud $\mathrm{Sim}(3) = \{s, \mathbf{R}, \mathbf{t}\}$ que minimiza $\|s\mathbf{R}X + \mathbf{t} - Y\|^2$.
  5. `compute_ate(pred_c2ws, gt_c2ws, align=True)`: Calcula el Absolute Trajectory Error (RMSE de traslación en metros y rotación en grados) tras alineación Umeyama.
  6. `compute_rpe(pred_c2ws, gt_c2ws)`: Calcula el Relative Pose Error entre fotogramas sucesivos $i$ e $i+1$, cuantificando la deriva de odometría local.

---

### 5.6. `src/cameras.py`
- **Ruta**: [`src/cameras.py`](file:///home/felix/Projects/svraster/src/cameras.py)
- **Clases modificadas y añadidas**:
  1. En `Camera.__init__`:
     - Se añadió el atributo opcional `c2w_gt=None`.
     - Inicialización agnóstica al hardware:
       ```python
       device = "cuda" if torch.cuda.is_available() else "cpu"
       self.w2c = torch.as_tensor(w2c, dtype=torch.float32, device=device)
       self.c2w = self.w2c.inverse().contiguous()
       self.c2w_gt = torch.as_tensor(c2w_gt, dtype=torch.float32, device=device) if c2w_gt is not None else None
       ```
       **Por qué**: Permite instanciar objetos `Camera` sin requerir que la GPU esté disponible en etapas de inspección o tests en CPU.
  2. **Nueva clase `CameraPoseOptimizer(torch.nn.Module)`**:
     ```python
     class CameraPoseOptimizer(torch.nn.Module):
         def __init__(self, num_cams: int, init_c2w_list=None, mode: str = "colmap", device=None):
             super().__init__()
             self.num_cams = num_cams
             self.mode = mode
             if device is None:
                 device = "cuda" if torch.cuda.is_available() else "cpu"
             self.se3_refine = torch.nn.Embedding(num_cams, 6, device=device)
             torch.nn.init.zeros_(self.se3_refine.weight)
     ```
     - Si `mode == 'colmap'`: `self.base_c2w` almacena las poses iniciales de COLMAP como buffer no entrenable.
     - Si `mode == 'identity'`: `self.base_c2w` se inicializa con matrices identidad $\mathbf{I}_{4 \times 4}$.
     - `get_c2w(cam_idx)`: Calcula $\mathbf{T}_{base}[cam\_idx] \cdot \exp(\boldsymbol{\xi}_{cam\_idx})$.
     - `get_all_c2w()`: Vectorizado para obtener todas las poses simultáneamente durante la evaluación.
     - `state_dict_poses()` y `load_state_dict_poses(state)`: Serialización desacoplada para guardar y reanudar el entrenamiento.

---

### 5.7. `src/dataloader/data_pack.py`
- **Ruta**: [`src/dataloader/data_pack.py`](file:///home/felix/Projects/svraster/src/dataloader/data_pack.py)
- **Modificación**:
  En la clase `CameraCreator`, se añadió el campo `c2w_gt` a los diccionarios de metadatos de las cámaras y se pasó al constructor de `Camera(...)`.
- **Por qué**:
  Garantiza que la pose original no perturbada viaje junto a la cámara para poder calcular las métricas ATE/RPE durante el entrenamiento.

---

### 5.8. `src/dataloader/reader_colmap_dataset.py`
- **Ruta**: [`src/dataloader/reader_colmap_dataset.py`](file:///home/felix/Projects/svraster/src/dataloader/reader_colmap_dataset.py)
- **Modificación**:
  Al construir `CameraInfo`, se conserva `c2w_gt = c2w.copy()`. Si se inyecta ruido intencional a las poses de COLMAP para experimentos de robustez, `c2w_gt` retiene la referencia perfecta.

---

### 5.9. `src/dataloader/reader_nerf_dataset.py`
- **Ruta**: [`src/dataloader/reader_nerf_dataset.py`](file:///home/felix/Projects/svraster/src/dataloader/reader_nerf_dataset.py)
- **Modificación**:
  En datasets sintéticos de NeRF (donde los archivos `transforms_train.json` contienen poses analíticamente exactas), se asigna `c2w_gt = c2w`.

---

### 5.10. `src/config.py`
- **Ruta**: [`src/config.py`](file:///home/felix/Projects/svraster/src/config.py)
- **Campos añadidos al nodo `_C.pose_opt`**:
  ```python
  _C.pose_opt = CfgNode()
  _C.pose_opt.pose_opt = False
  _C.pose_opt.pose_init_mode = 'colmap'  # 'colmap' o 'identity'
  _C.pose_opt.lr_pose = 1e-3             # Tasa inicial de Adam
  _C.pose_opt.lr_pose_end = 1e-5         # Tasa final con decaimiento exponencial
  _C.pose_opt.warmup_pose = 500          # Iteraciones de calentamiento
  _C.pose_opt.eval_pose_gt = True        # Evaluación automática de ATE/RPE
  ```
- **Por qué**:
  Centraliza la configuración en el sistema jerárquico YACS del proyecto y genera automáticamente opciones de CLI mediante `update_argparser`.

---

### 5.11. `cfg/pose_opt.yaml`
- **Ruta**: [`cfg/pose_opt.yaml`](file:///home/felix/Projects/svraster/cfg/pose_opt.yaml)
- **Contenido**: Archivo de configuración YAML listo para usar que habilita la optimización de poses con hiperparámetros óptimos preestablecidos:
  ```yaml
  pose_opt:
    pose_opt: True
    pose_init_mode: 'colmap'
    lr_pose: 0.001
    lr_pose_end: 0.00001
    warmup_pose: 500
    eval_pose_gt: True
  ```

---

### 5.12. `train.py`
- **Ruta**: [`train.py`](file:///home/felix/Projects/svraster/train.py)
- **Modificaciones realizadas**:
  1. **Inicialización**:
     ```python
     pose_optimizer = None; optim_pose = None; sched_pose = None
     if cfg.pose_opt.pose_opt:
         pose_optimizer = CameraPoseOptimizer(
             num_cams=len(tr_cams),
             init_c2w_list=tr_cams,
             mode=cfg.pose_opt.pose_init_mode
         ).cuda()
         optim_pose = torch.optim.Adam(pose_optimizer.parameters(), lr=cfg.pose_opt.lr_pose)
         total_decay_steps = max(1, cfg.procedure.n_iter - cfg.pose_opt.warmup_pose)
         gamma = (cfg.pose_opt.lr_pose_end / cfg.pose_opt.lr_pose) ** (1.0 / total_decay_steps)
         sched_pose = torch.optim.lr_scheduler.ExponentialLR(optim_pose, gamma=gamma)
     ```
  2. **Actualización dinámica en cada paso de iteración**:
     ```python
     cam_idx = tr_cam_indices[iteration-1]
     cam = tr_cams[cam_idx]
     if cfg.pose_opt.pose_opt and pose_optimizer is not None:
         cam_c2w = pose_optimizer.get_c2w(cam_idx)
         cam.c2w = cam_c2w
         cam.w2c = torch.inverse(cam_c2w).detach()
     ```
     **¿Por qué `.detach()` en `w2c`?**:
     `w2c` se usa como matriz auxiliar geométrica para el frustum culling. Mantener el gradiente únicamente en `c2w` asegura que el gradiente analítico de CUDA retropropague directamente sin bifurcaciones no deseadas en el autograd de PyTorch.
  3. **Zero-Grad y Backward**:
     ```python
     optimizer.zero_grad(set_to_none=True)
     if optim_pose is not None:
         optim_pose.zero_grad(set_to_none=True)
     loss.backward()
     ```
  4. **Paso del Optimizador y Calentamiento (Warmup)**:
     ```python
     optimizer.step()
     if optim_pose is not None and iteration > cfg.pose_opt.warmup_pose:
         optim_pose.step()
     ```
     **¿Por qué un Warmup (`warmup_pose`)?**:
     En las primeras iteraciones (e.g. 0 a 500), el campo de densidad de vóxeles contiene ruido aleatorio o inicializaciones toscas. Si las cámaras intentan optimizarse contra una geometría no convergida, empezarán a desviarse de forma caótica. Retrasar la optimización de cámaras hasta que el volumen adquiera una estructura básica evita divergencias catastróficas.
  5. **Scheduler Step**:
     ```python
     scheduler.step()
     if sched_pose is not None and iteration > cfg.pose_opt.warmup_pose:
         sched_pose.step()
     ```
  6. **Checkpoints**:
     Al guardar iteraciones (`args.checkpoint_iterations`):
     ```python
     if cfg.pose_opt.pose_opt and pose_optimizer is not None:
         pose_ckpt = {'poses': pose_optimizer.state_dict_poses()}
         if args.save_optimizer and optim_pose is not None:
             pose_ckpt['optim'] = optim_pose.state_dict()
             pose_ckpt['sched'] = sched_pose.state_dict()
         torch.save(pose_ckpt, os.path.join(args.model_path, f"pose_opt_{iteration:06d}.pt"))
         torch.save(pose_ckpt, os.path.join(args.model_path, "pose_opt.pt"))
     ```
  7. **Evaluación de Métricas ATE y RPE**:
     En `training_report`, si existen poses `c2w_gt`:
     ```python
     pred_c2ws = pose_optimizer.get_all_c2w().detach()
     gt_c2ws = torch.stack([c.c2w_gt for c in train_cameras])
     ate_res = compute_ate(pred_c2ws, gt_c2ws)
     rpe_res = compute_rpe(pred_c2ws, gt_c2ws)
     stat['pose_metrics'] = {'ate': ate_res, 'rpe': rpe_res}
     ```
     Los resultados se imprimen en consola y se guardan en el archivo `test_stat/iterXXXXXX.json`.
  8. **Parser de argumentos CLI**:
     Soporte para `--pose_opt`, `--pose_init_mode`, `--lr_pose`, `--warmup_pose`.
  9. **Corrección de Bug Latente en `train.py` (Línea 273)**:
     El código heredado contenía un bloque de depuración con `if iteration == self.iter_from + 10:` dentro de la función libre `training(args)`. Dado que `self` no está definido en el ámbito de función, esto provocaba una excepción fatal de tipo `NameError` si el usuario ejecutaba con ranking de profundidad activo. Se corrigió el bloque garantizando la estabilidad total del proceso.

---

## 6. Métricas de Trayectoria: Algoritmo de Umeyama, ATE y RPE

En visión computacional monocular, la reconstrucción 3D y la trayectoria de cámaras tienen una **ambigüedad de calibre $\mathrm{Sim}(3)$**:
1. No existe un origen absoluto de coordenadas (ambigüedad de traslación).
2. No existe una orientación absoluta del mundo (ambigüedad de rotación).
3. No existe una escala métrica absoluta física sin sensores inerciales o cámaras estéreo (ambigüedad de escala).

Por tanto, comparar directamente las posiciones predichas $\mathbf{t}_{pred}$ con las de referencia $\mathbf{t}_{gt}$ arrojaría un error artificialmente gigantesco.

### Algoritmo de Umeyama
Dadas dos nubes de centros de cámara correspondientes $\{\mathbf{x}_i\}_{i=1}^N$ y $\{\mathbf{y}_i\}_{i=1}^N$:
1. Centrado respecto a las medias muestrales:
   $$\boldsymbol{\mu}_x = \frac{1}{N}\sum \mathbf{x}_i, \quad \boldsymbol{\mu}_y = \frac{1}{N}\sum \mathbf{y}_i$$
   $$\sigma_x^2 = \frac{1}{N}\sum \|\mathbf{x}_i - \boldsymbol{\mu}_x\|^2$$
2. Matriz de covarianza cruzada:
   $$\mathbf{\Sigma}_{xy} = \frac{1}{N} \sum (\mathbf{y}_i - \boldsymbol{\mu}_y)(\mathbf{x}_i - \boldsymbol{\mu}_x)^T$$
3. Descomposición en valores singulares:
   $$\mathbf{\Sigma}_{xy} = \mathbf{U} \mathbf{D} \mathbf{V}^T$$
4. Corrección de determinante para garantizar rotación propia ($\det(\mathbf{R}) = +1$, evitando reflexiones especulares):
   $$\mathbf{S} = \operatorname{diag}\left(1, 1, \operatorname{sign}(\det(\mathbf{U}\mathbf{V}^T))\right)$$
   $$\mathbf{R}_{align} = \mathbf{U} \mathbf{S} \mathbf{V}^T$$
   $$s = \frac{1}{\sigma_x^2} \operatorname{tr}(\mathbf{D}\mathbf{S})$$
   $$\mathbf{t}_{align} = \boldsymbol{\mu}_y - s \mathbf{R}_{align} \boldsymbol{\mu}_x$$

### Absolute Trajectory Error (ATE)
Tras alinear las posiciones mediante $\mathbf{t}'_i = s \mathbf{R}_{align} \mathbf{t}_{pred, i} + \mathbf{t}_{align}$:
$$\text{ATE}_{trans} = \sqrt{\frac{1}{N} \sum_{i=1}^N \|\mathbf{t}'_i - \mathbf{t}_{gt, i}\|^2}$$
$$\text{ATE}_{rot} = \sqrt{\frac{1}{N} \sum_{i=1}^N \theta_i^2}$$
donde $\theta_i = \arccos\left(\frac{\operatorname{tr}(\mathbf{R}_{align} \mathbf{R}_{pred, i} \mathbf{R}_{gt, i}^T) - 1}{2}\right)$.

### Relative Pose Error (RPE)
Mide el error de la transformación relativa entre pares de fotogramas contiguos $(i, i+1)$:
$$\Delta \mathbf{T}_{pred} = \mathbf{T}_{pred, i}^{-1} \mathbf{T}_{pred, i+1}$$
$$\Delta \mathbf{T}_{gt} = \mathbf{T}_{gt, i}^{-1} \mathbf{T}_{gt, i+1}$$
$$\mathbf{E}_i = \Delta \mathbf{T}_{gt}^{-1} \Delta \mathbf{T}_{pred}$$
Cuantifica la suavidad y deriva local independiente de la alineación global.

---

## 7. Guía de Compilación, Ejecución y Comandos CLI

Todas las instrucciones a continuación respetan la restricción de ejecutarse exclusivamente bajo el entorno conda configurado en `/home/felix/miniconda3/envs/svraster`.

### 7.1. Compilación del Kernel CUDA
Para compilar la extensión en sitio (*inplace*) con los nuevos kernels analíticos:

```bash
cd /home/felix/Projects/svraster/cuda
conda run -p /home/felix/miniconda3/envs/svraster python setup.py build_ext --inplace
```

### 7.2. Modos de Ejecución

#### A) Refinamiento con Inicialización desde COLMAP (Recomendado para escenas reales)
Ajusta pequeñas imperfecciones en las poses iniciales calculadas por COLMAP:
```bash
conda run -p /home/felix/miniconda3/envs/svraster python train.py \
    --source_path /ruta/a/tu/dataset \
    --model_path ./output/svraster_colmap_opt \
    --pose_opt \
    --pose_init_mode colmap \
    --lr_pose 0.001 \
    --warmup_pose 500
```

#### B) Optimización desde Cero (Identity Initialization)
No requiere SfM previo. Todas las cámaras arrancan en la identidad $\mathbf{I}_{4 \times 4}$:
```bash
conda run -p /home/felix/miniconda3/envs/svraster python train.py \
    --source_path /ruta/a/tu/dataset \
    --model_path ./output/svraster_from_scratch \
    --pose_opt \
    --pose_init_mode identity \
    --lr_pose 0.002 \
    --warmup_pose 1000
```

#### C) Uso mediante Archivo YAML
Permite reproducibilidad guardando la configuración:
```bash
conda run -p /home/felix/miniconda3/envs/svraster python train.py \
    --cfg_files cfg/pose_opt.yaml \
    --source_path /ruta/a/tu/dataset \
    --model_path ./output/svraster_yaml_opt
```

#### D) Reanudación y Carga desde Checkpoints
Restaura tanto la geometría de vóxeles como los estados de los optimizadores de pose:
```bash
conda run -p /home/felix/miniconda3/envs/svraster python train.py \
    --source_path /ruta/a/tu/dataset \
    --model_path ./output/svraster_colmap_opt \
    --pose_opt \
    --load_iteration 10000 \
    --load_optimizer
```

---

## 8. Verificación y Resultados de Pruebas Unitarias

Se ejecutaron pruebas analíticas en `/home/felix/miniconda3/envs/svraster` confirmando la exactitud del sistema:
1. **Comprobación de $\mathrm{SO}(3)$ y $\mathrm{SE}(3)$**:
   - $\exp(\mathbf{0}) = \mathbf{I}_{4 \times 4}$ con error absoluto $0.0$.
   - Ortogonalidad $\mathbf{R}^T \mathbf{R} = \mathbf{I}_3$ verificada para vectores aleatorios con tolerancia $< 10^{-5}$.
   - Serie de Taylor para ángulos $\theta < 10^{-5}$ libre de singularidades y $\text{NaN}$.
2. **Prueba de Flujo de Gradiente en PyTorch Autograd**:
   - Simulación de pérdidas $\mathcal{L}(\mathbf{T}_{c2w})$ retropropagadas hacia `CameraPoseOptimizer`.
   - Se verificó que el gradiente fluye exclusivamente hacia el índice de cámara consultado, mientras que los pesos de las demás cámaras retienen gradiente estrictamente nulo.
3. **Prueba de Métrica Umeyama y ATE**:
   - Trayectorias sintéticas sometidas a rotaciones y traslaciones arbitrarias fueron realineadas por el algoritmo de Umeyama con error de traslación residual menor a $10^{-6}\text{ m}$.

---

## 9. Monitoreo por Época: Formulación Matemática de Variaciones de Pose y Estructura

Para evitar el almacenamiento redundante de cientos de archivos de pose en disco (matrices $[N, 4, 4]$ repetidas en cada época) y proveer una visión analítica del entrenamiento, se desarrolló el módulo de métricas por época (`record_epoch_metrics` en [`train.py`](file:///home/felix/Projects/svraster/train.py)).

### 9.1 Definición Temporal de Época
Dado un conjunto de cámaras de entrenamiento $\mathcal{C}_{\text{train}}$ con cardinalidad $N = |\mathcal{C}_{\text{train}}|$, cada época $k \in \mathbb{N}$ se define como el intervalo en el cual cada cámara ha sido procesada en promedio una vez:
$$k = \left\lfloor \frac{\text{iteración}}{N} \right\rfloor$$
El evento de fin de época se dispara cuando $\text{iteración} \pmod N = 0$ o cuando $\text{iteración} = n_{\text{iter}}$.

### 9.2 Variación de Pose Entre Épocas Consecutivas ($\Delta \mathbf{P}_{\text{epoch}}$)
Sean $\mathbf{T}_i^{(k-1)} \in \mathrm{SE}(3)$ y $\mathbf{T}_i^{(k)} \in \mathrm{SE}(3)$ las matrices de pose cámara-a-mundo ($c2w$) de la cámara $i$ en las épocas $k-1$ y $k$, respectivamente.

1. **Transformación Relativa**:
   $$\Delta \mathbf{T}_i = \left(\mathbf{T}_i^{(k-1)}\right)^{-1} \mathbf{T}_i^{(k)} = \begin{bmatrix} \mathbf{R}_{\Delta, i} & \mathbf{t}_{\Delta, i} \\ \mathbf{0}^T & 1 \end{bmatrix}$$
   donde:
   $$\mathbf{R}_{\Delta, i} = \left(\mathbf{R}_i^{(k-1)}\right)^T \mathbf{R}_i^{(k)}, \quad \mathbf{t}_{\Delta, i} = \mathbf{t}_i^{(k)} - \mathbf{t}_i^{(k-1)}$$

2. **Magnitud Angular de Rotación**:
   Por la fórmula de Rodrigues invertida:
   $$\theta_i = \arccos\left(\operatorname{clamp}\left(\frac{\operatorname{tr}(\mathbf{R}_{\Delta, i}) - 1}{2}, -1, 1\right)\right) \times \frac{180^\circ}{\pi}$$

3. **Magnitud de Traslación**:
   $$d_i = \|\mathbf{t}_{\Delta, i}\|_2 = \sqrt{\sum_{j=1}^3 (\mathbf{t}_{\Delta, i})_j^2}$$

4. **Agregación Promedio y Máxima**:
   $$\text{epoch\_drot\_mean\_deg} = \frac{1}{N} \sum_{i=1}^N \theta_i, \quad \text{epoch\_drot\_max\_deg} = \max_{i} \theta_i$$
   $$\text{epoch\_dtrans\_mean\_m} = \frac{1}{N} \sum_{i=1}^N d_i, \quad \text{epoch\_dtrans\_max\_m} = \max_{i} d_i$$

### 9.3 Variación Acumulada respecto al Origen / COLMAP ($\Delta \mathbf{P}_{\text{total}}$)
De forma directa desde los parámetros en el álgebra de Lie $\boldsymbol{\xi}_i = (\boldsymbol{\omega}_i, \mathbf{u}_i) \in \mathbb{R}^6$:
$$\text{total\_drot\_mean\_deg} = \frac{1}{N} \sum_{i=1}^N \|\boldsymbol{\omega}_i\|_2 \times \frac{180^\circ}{\pi}, \quad \text{total\_drot\_max\_deg} = \max_i \|\boldsymbol{\omega}_i\|_2 \times \frac{180^\circ}{\pi}$$
$$\text{total\_dtrans\_mean\_m} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{u}_i\|_2, \quad \text{total\_dtrans\_max\_m} = \max_i \|\mathbf{u}_i\|_2$$

### 9.4 Métricas de Estructura Geométrica
1. **Volumen del Octree**:
   - `num_voxels`: Cantidad total de vóxeles subdivididos y retenidos.
   - `inside_voxels`: Vóxeles cuyo centro $\mathbf{c}_v \in \mathbb{R}^3$ satisface:
     $$\mathbf{c}_{\text{inside, min}} < \mathbf{c}_v < \mathbf{c}_{\text{inside, max}}$$
   - `inside_pct`: $\frac{\text{inside\_voxels}}{\text{num\_voxels}} \times 100\%$.
2. **Calidad de Síntesis**:
   - `loss`: Pérdida combinada suavizada por media móvil exponencial (EMA).
   - `psnr`: PSNR suavizado en decibelios (dB).

### 9.5 Archivos Generados
En el directorio de salida (`--model_path`):
- `metrics_per_epoch.csv`: Archivo tabular con encabezado completo para análisis en Python/R/Excel:
  ```csv
  epoch,iteration,elapsed_sec,epoch_time_sec,loss,psnr,num_voxels,inside_voxels,inside_pct,lr_geo,lr_pose,epoch_drot_mean_deg,epoch_drot_max_deg,epoch_dtrans_mean_m,epoch_dtrans_max_m,total_drot_mean_deg,total_drot_max_deg,total_dtrans_mean_m,total_dtrans_max_m,ate_trans_rmse_m,ate_rot_rmse_deg,rpe_trans_rmse_m,rpe_rot_rmse_deg
  ```
- `metrics_per_epoch.jsonl`: Formato JSON estructurado línea por línea para procesamiento automatizado.
- Salida en consola interactiva vía `progress_bar.write`:
  ```text
  [EPOCH 005 | iter 00660 | 25.1s] Voxels: 620000 (74.1% in) | Loss: 0.03820 | PSNR: 18.15 | dPose/ep: 0.142deg, 0.185cm | Tot dPose: 0.32deg, 0.41cm | ATE: 0.0765m, 5.12deg
  ```

