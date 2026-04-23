# 🏛️ Landmark Classifier — Clasificación de Landmarks con CNN

Proyecto final del Módulo 6 (Redes Neuronales) de la Maestría en Ciencia de Datos e IA Aplicada. Sistema de clasificación multiclase que identifica **50 landmarks** (puntos de referencia mundiales) a partir de imágenes, desarrollado con PyTorch.

## 📌 Descripción

Se implementan y comparan dos enfoques:

1. **CNN diseñada desde cero** — `LandmarkCNN`, 4 bloques convolucionales, ~547 K parámetros.
2. **Transfer Learning con ResNet50** preentrenada en ImageNet, con fine-tuning del último bloque convolucional.

Adicionalmente se incluye una **aplicación de predicción** con función `predict_landmarks(img, k)` y una interfaz web interactiva construida con Gradio.

## 📈 Resultados obtenidos

| Modelo | Test Accuracy | Umbral | Estado |
|--------|--------------:|-------:|:------:|
| CNN From Scratch (30 ép. + 11 de continuación con early stopping) | **41.92%** | ≥ 40% | ✅ |
| Transfer Learning — ResNet50 (15 feature-extract + 10 fine-tune) | **82.72%** | ≥ 70% | ✅ (bono +75% ✅) |

**Validación con imágenes propias:** 3/4 correctas en top-1, 4/4 en top-3.

Detalle completo del proyecto en [`report.pdf`](report.pdf) (fuente en [`report/resumen_proyecto.md`](report/resumen_proyecto.md)).

## 🛠️ Stack tecnológico

- **Framework:** PyTorch 2.5.1 + TorchVision 0.20.1 (build CUDA 12.1)
- **Python:** 3.11.9
- **Hardware de referencia:** NVIDIA RTX 2050, 4 GB VRAM (CUDA recomendado; corre en CPU, mucho más lento)
- **Interfaz:** Gradio 6.x
- **Otros:** Jupyter, matplotlib, seaborn, pandas, numpy, Pillow, scikit-learn, tqdm

## 📁 Estructura del repositorio

```
landmark-classifier/
├── README.md
├── requirements.txt
├── test_gpu.py                  # Verifica PyTorch + CUDA + GPU
├── report.pdf                   # Resumen del proyecto en PDF
├── exploracion_y_preproceso.ipynb    # Fase 1: EDA + DataLoaders
├── cnn_from_scratch.ipynb            # Fase 2: CNN propia (30 ép. + continuación)
├── transfer_learning.ipynb           # Fase 3: ResNet50 transfer learning
├── app.ipynb                         # Fase 4: inferencia + Gradio
├── src/
│   ├── data.py                  # get_transforms, get_dataloaders
│   ├── model.py                 # LandmarkCNN, get_transfer_model
│   ├── train.py                 # train_one_epoch, validate, train_model
│   └── predictor.py             # predict_landmarks para inferencia
├── models/                      # Checkpoints .pt + versiones TorchScript
├── report/                      # Gráficos generados + resumen_proyecto.md (fuente del PDF)
├── data/                        # Dataset (NO incluido en el repo — 50 clases)
│   ├── train/
│   └── test/
└── images_test/                 # Imágenes propias para Fase 4 (opcional)
```

## 🚀 Instalación

Requiere **Python 3.11** específicamente. Si en tu sistema el comando `python` apunta a otra versión (ej. 3.14), usá el launcher de Windows `py -3.11` o la ruta completa al ejecutable de 3.11.

### 1. Clonar el repositorio

```powershell
git clone <URL-del-repo>
cd landmark-classifier
```

### 2. Crear el entorno virtual con Python 3.11

```powershell
# Opción A — si tenés python3.11 en PATH:
python3.11 -m venv venv

# Opción B — launcher de Windows (recomendada):
py -3.11 -m venv venv

# Opción C — ruta completa al ejecutable:
C:\ruta\a\python3.11\python.exe -m venv venv
```

### 3. Activar el entorno

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate
```

> Si PowerShell bloquea la activación con un error de política de scripts:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

Una vez activo el venv, `python` ya apunta a 3.11 — verificalo con `python --version`.

### 4. Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

El `requirements.txt` incluye `--extra-index-url` con el índice de PyTorch para CUDA 12.1; si no tenés GPU NVIDIA, PyTorch caerá automáticamente a la versión CPU.

### 5. Verificar la instalación

```powershell
python test_gpu.py
```

Debe imprimir `CUDA disponible: True` y detectar tu GPU. Si aparece `CUDA: False` y tenés una GPU NVIDIA, reinstalá PyTorch con:

```powershell
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 6. Registrar el kernel de Jupyter

Para que los notebooks encuentren el entorno:

```powershell
python -m ipykernel install --user --name=landmark-classifier --display-name "Python (landmark-classifier)"
```

Verificalo con `jupyter kernelspec list`.

### 7. Descargar el dataset

Colocar el dataset en `data/` con la siguiente estructura:

```
data/
├── train/
│   ├── 00.Haleakala_National_Park/
│   ├── 01.Mount_Rainier_National_Park/
│   └── ... (50 carpetas, ~100 imágenes cada una)
└── test/
    ├── 00.Haleakala_National_Park/
    └── ... (50 carpetas, 25 imágenes cada una)
```

## 📊 Ejecución del proyecto

### Notebooks en orden

Los notebooks viven en la raíz del repo y usan rutas relativas (`data/`, `models/`, `report/`), por lo que deben ejecutarse con la raíz del repo como directorio de trabajo (`jupyter notebook` o `jupyter lab` lanzado desde ahí).

1. **`exploracion_y_preproceso.ipynb`** — Exploración del dataset, visualizaciones, creación de DataLoaders. Genera 3 PNGs en `report/`. Rápido.
2. **`cnn_from_scratch.ipynb`** — CNN propia (Fase 2 + continuación consolidadas en un solo notebook): 30 épocas iniciales + hasta 15 épocas adicionales con early stopping manual desde checkpoint. Guarda `models/cnn_scratch_best.pt` y su versión TorchScript. Duración típica: ~22 min en RTX 2050.
3. **`transfer_learning.ipynb`** — ResNet50 en dos pasos (feature extraction 15 épocas + fine-tuning de `layer4` 10 épocas). Guarda `models/resnet50_finetuned_best.pt` y `resnet50_torchscript.pt`. Duración: ~18 min.
4. **`app.ipynb`** — Inferencia sobre imágenes propias (si poblás `images_test/`) + construcción de la interfaz Gradio.

### Ejecutar notebooks desde la línea de comandos

Útil para reproducir en batch:

```powershell
jupyter nbconvert --to notebook --execute cnn_from_scratch.ipynb --inplace
```

### Probar imágenes propias (Fase 4)

1. Copiar ≥4 imágenes `.jpg` o `.png` a `images_test/`.
2. Abrir `app.ipynb` y ejecutar la celda de predicciones.
3. El resultado se guarda en `report/predicciones_propias.png`.

### Lanzar la interfaz Gradio

La última celda del notebook `app.ipynb` construye la interfaz pero el `demo.launch(...)` está protegido por la variable de entorno `LAUNCH_GRADIO` (para no bloquear ejecuciones batch).

**Desde Jupyter** — abrir el notebook, ejecutar todas las celdas, y en la última correr manualmente:

```python
demo.launch(share=False, inbrowser=True)
```

**Desde la terminal** — setear la variable antes de lanzar Jupyter:

```powershell
# PowerShell
$env:LAUNCH_GRADIO=1; jupyter notebook

# Git Bash
LAUNCH_GRADIO=1 jupyter notebook
```

La app queda disponible en `http://localhost:7860`.

## 🧩 Usar el código fuera de los notebooks

Todo el código es modular. Ejemplo de inferencia directa:

```python
import sys; sys.path.append("src")
from src.predictor import load_model, predict_landmarks
from src.data import get_dataloaders

_, _, _, classes = get_dataloaders(data_dir="data", num_workers=0)
model = load_model("models/resnet50_torchscript.pt", device="cuda")
preds = predict_landmarks("mi_foto.jpg", model, classes, k=3, device="cuda")
# [('16.Eiffel_Tower', 0.876), ('31.Washington_Monument', 0.080), ...]
```

## 📝 Autor

Klaus Uribe — Maestría en Ciencia de Datos e IA Aplicada

## 📄 Licencia

Proyecto académico con fines educativos.
