# Clasificación de Razas de Mascotas con Transfer Learning

Este repositorio contiene el código y los resultados del proyecto individual para el módulo de Deep Learning de **PhawAI + TaReCDa 2025**. El objetivo es realizar un estudio comparativo de tres modelos de Redes Neuronales Convolucionales (MobileNetV2, EfficientNetB0 y ResNet50) para la clasificación de 37 razas de mascotas, utilizando la técnica de Transfer Learning.

---

### Tabla de Contenidos
1.  [Ruta Elegida y Dataset](#1-ruta-elegida-y-dataset)
2.  [Cómo Ejecutar el Proyecto](#2-cómo-ejecutar-el-proyecto)
3.  [Cómo Entrenar y Evaluar](#3-cómo-entrenar-y-evaluar)
4.  [Cómo Generar los Resultados](#4-cómo-generar-los-resultados)

---

### 1. Ruta Elegida y Dataset

*   **Ruta Elegida:**
    *   Clasificación con modelos preentrenados (Transfer Learning).

*   **Dataset:**
    *   **Fuente:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), cargado directamente a través de la API de [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet).
    *   **Licencia:** [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
    *   **Descripción:** El dataset contiene aproximadamente 7,390 imágenes de 37 razas de perros y gatos, con cerca de 200 imágenes por clase. Fue particionado en 70% para entrenamiento, 15% para validación y 15% para pruebas.

---

### 2. Cómo Ejecutar el Proyecto

El proyecto está diseñado para ser ejecutado en **Google Colab** para aprovechar las GPUs gratuitas.

#### Prerrequisitos
Las librerías necesarias se encuentran listadas en la sección `1.2 Importación de Módulos` del notebook. Las principales son:
*   `tensorflow==2.15` (o superior)
*   `tensorflow_datasets`
*   `numpy`
*   `pandas`
*   `seaborn`
*   `matplotlib`
*   `scikit-learn`

#### Pasos para la Ejecución
1.  **Abrir en Google Colab:**
    *   Abre el notebook `Proyecto_Clasificacion.ipynb` en Google Colab.

2.  **Seleccionar Entorno de Ejecución con GPU:**
    *   En el menú de Colab, ve a `Entorno de ejecución` -> `Cambiar tipo de entorno de ejecución`.
    *   Selecciona **T4 GPU** como acelerador de hardware. Esto es crucial para un entrenamiento rápido.

3.  **Montar Google Drive:**
    *   Ejecuta la celda `1.3 Conexión con Google Drive` para montar tu unidad. Esto es necesario para que el notebook pueda guardar los modelos entrenados y los resultados (figuras y tablas) en tu cuenta de Drive.

4.  **Ejecutar Todas las Celdas:**
    *   Ve a `Entorno de ejecución` -> `Ejecutar todas`.
    *   El notebook ejecutará secuencialmente todo el proceso: configuración, carga de datos, preprocesamiento, entrenamiento de los tres modelos, evaluación y generación de todos los resultados.
    *   **Nota:** La ejecución completa puede tardar un tiempo considerable, especialmente las celdas de entrenamiento y evaluación.

---

### 3. Cómo Entrenar y Evaluar

El entrenamiento y la evaluación se gestionan de forma automática al ejecutar el notebook. Los pasos principales que sigue el código son:

1.  **Configuración Global (Sección 1.4):** Se establecen la semilla de aleatoriedad para la reproducibilidad, los hiperparámetros globales (tamaño de imagen, batch size) y la política de `mixed precision` para acelerar el entrenamiento.

2.  **Carga y Preprocesamiento de Datos (Sección 2):** El dataset se carga desde TFDS. Se crea una pipeline de `tf.data` que aplica el redimensionamiento, el aumento de datos (solo para entrenamiento) y la normalización específica de cada arquitectura.

3.  **Entrenamiento de los Modelos (Sección 3):**
    *   El script itera sobre los tres modelos: `MobileNetV2`, `EfficientNetB0` y `ResNet50`.
    *   Para cada modelo, se ejecuta un protocolo de dos fases:
        1.  **Transfer Learning (10 épocas):** Se entrena únicamente la cabeza clasificadora con el `backbone` congelado.
        2.  **Fine-Tuning (10 épocas):** Se descongelan las capas superiores del `backbone` y se continúa el entrenamiento con una tasa de aprendizaje más baja.

4.  **Evaluación Comprensiva (Sección 4):**
    *   Una vez entrenados los tres modelos, se evalúa cada uno en los conjuntos de entrenamiento y prueba.
    *   Se calculan las métricas `Accuracy`, `F1-Macro` y `Precision Macro`.

---

### 4. Cómo Generar los Resultados

La generación de todas las tablas y gráficos para el informe se realiza automáticamente en las últimas secciones del notebook, después de la evaluación. Los resultados se guardan en la carpeta `/results/` dentro de la ruta de tu proyecto en Google Drive.

*   **Tabla Comparativa (Sección 4.3):**
    *   La función `create_detailed_comparative_table` genera la tabla principal con los resultados de las métricas.
    *   El resultado se guarda como `comparative_table_detailed.csv` y también se muestra en el notebook.

*   **Gráficos de Overfitting y Comparación de Métricas (Secciones 4.4 y 4.6):**
    *   Las funciones `plot_detailed_train_test_bars` y `plot_metrics_comparison` generan los gráficos de barras comparativos.
    *   Se guardan como `overfitting_analysis_detailed_bars.png` y `metrics_comparison_bars.png`.

*   **Curvas de Entrenamiento y Matrices de Confusión (Secciones 4.6 y 4.7):**
    *   El script genera y guarda las curvas de `Accuracy` y `Loss` para cada modelo (ej. `curves_mobilenetv2.png`).
    *   También se generan las matrices de confusión como `confusion_matrices_heatmap.png` y se exportan los resúmenes de errores.

---

---

### 5. Licencia

El **código fuente** de este proyecto se distribuye bajo la **Licencia MIT**. Para más detalles, consulta el archivo `LICENSE`.

El **dataset** utilizado, Oxford-IIIT Pet, es propiedad de sus respectivos autores y se distribuye bajo la licencia [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). De acuerdo con la cláusula "ShareAlike", todos los trabajos derivados de este dataset, incluyendo los **modelos entrenados** y los **resultados visuales** (gráficos, tablas, etc.) presentados en este repositorio, también se consideran distribuidos bajo la misma licencia CC BY-SA 4.0.
