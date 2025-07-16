# 🧬 Evaluación de Modelos ML para Valoración Nutricional Antropométrica

Este proyecto de Streamlit proporciona una aplicación interactiva para la **evaluación y comparación de modelos de Machine Learning** aplicados a la valoración nutricional antropométrica. La herramienta permite cargar datos de pacientes, preprocesarlos, entrenar diferentes modelos de regresión y visualizar sus métricas de rendimiento, tiempos de entrenamiento y resultados de pruebas estadísticas. Además, genera reportes PDF detallados para cada variable objetivo evaluada.

## Características Principales

* **Carga de Datos Simplificada**: Carga automáticamente el archivo `datos_pacientes2.csv` para un inicio rápido.
* **Preprocesamiento Automático**: Incluye codificación de variables categóricas con `LabelEncoder` y manejo de valores nulos, asegurando que los datos estén listos para el entrenamiento del modelo.
* **Evaluación de Múltiples Variables Objetivo**: Permite la evaluación de modelos para `Valoracion_Talla_Edad` y `Valoracion_IMC_Talla`.
* **Modelos de Regresión Implementados**:
    * **Regresión Lineal**: Un modelo fundamental para establecer relaciones lineales.
    * **Random Forest Regressor**: Un potente modelo de ensamble basado en árboles de decisión.
    * **XGBoost Regressor**: Un algoritmo de *gradient boosting* optimizado para el rendimiento y la velocidad.
* **Métricas de Rendimiento Clave**: Calcula y muestra MSE, RMSE, MAE y R² para una evaluación completa del rendimiento del modelo.
* **Análisis de Tiempos de Entrenamiento**: Mide y compara el tiempo que tarda cada modelo en entrenarse, crucial para la eficiencia.
* **Pruebas Estadísticas Inferenciales**: Realiza pruebas de normalidad de residuos (Shapiro-Wilk o Kolmogorov-Smirnov) para evaluar la validez de los supuestos del modelo.
* **Reportes PDF Detallados**: Genera reportes descargables para cada variable objetivo, incluyendo información del equipo, detalles del dataset, métricas, tiempos y resultados de pruebas estadísticas. También incorpora visualizaciones adicionales como matrices de confusión y gráficos de rendimiento (curvas de precisión-recall, ROC y calibración), siempre que las imágenes estén disponibles en las rutas esperadas.
* **Soporte Multilingüe**: La interfaz de usuario está disponible en español e inglés para una mayor accesibilidad.

## Estructura de Carpetas Esperada

Para que la generación de reportes PDF y la inclusión de gráficos funcionen correctamente, asegúrate de que tu proyecto tenga la siguiente estructura de carpetas:

.
├── datos_pacientes2.csv
├── app.py                  (Tu código de Streamlit)
├── confusion_matrices/
│   ├── confusion_matrix_talla_edad_regresion_lineal.png
│   ├── confusion_matrix_talla_edad_random_forest.png
│   ├── confusion_matrix_talla_edad_xgboost.png
│   ├── confusion_matrix_imc_talla_regresion_lineal.png
│   ├── confusion_matrix_imc_talla_random_forest.png
│   └── confusion_matrix_imc_talla_xgboost.png
└── Graficas/
├── RegresionLinealMulti/
│   ├── talla_edad_precision_recall_regresionlinealmulti.png
│   ├── talla_edad_roc_regresionlinealmulti.png
│   ├── talla_edad_calibration_regresionlinealmulti.png
│   ├── imc_talla_precision_recall_regresionlinealmulti.png
│   ├── imc_talla_roc_regresionlinealmulti.png
│   └── imc_talla_calibration_regresionlinealmulti.png
├── RandomForest/
│   ├── talla_edad_precision_recall_randomforest.png
│   ├── talla_edad_roc_randomforest.png
│   ├── talla_edad_calibration_randomforest.png
│   ├── imc_talla_precision_recall_randomforest.png
│   ├── imc_talla_roc_randomforest.png
│   └── imc_talla_calibration_randomforest.png
└── XGBoost/
├── talla_edad_precision_recall_xgboost.png
├── talla_edad_roc_xgboost.png
├── talla_edad_calibration_xgboost.png
├── imc_talla_precision_recall_xgboost.png
├── imc_talla_roc_xgboost.png
└── imc_talla_calibration_xgboost.png

**Nota**: Los nombres de los archivos de imagen dentro de `confusion_matrices/` y `Graficas/` deben seguir el formato especificado en el código para ser correctamente identificados y mostrados en los reportes PDF. Si tus nombres de archivo son diferentes, necesitarás ajustarlos en la función `add_images_to_pdf` del script `app.py`.

## Instalación y Ejecución

Para ejecutar esta aplicación en tu entorno local, sigue los siguientes pasos:

1.  **Clona el repositorio:**

    ```bash
    git clone <URL_DE_TU_REPOSITORIO>
    cd <nombre_de_tu_repositorio>
    ```

2.  **Crea un entorno virtual (opcional pero recomendado):**

    ```bash
    python -m venv venv
    # En Windows
    .\venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```

3.  **Instala las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

    (Asegúrate de tener un archivo `requirements.txt` con todas las bibliotecas listadas en tu `app.py`, como `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `reportlab`, `scipy`).

4.  **Coloca tu archivo de datos:**

    Asegúrate de que `datos_pacientes2.csv` esté en la misma carpeta que `app.py`.

5.  **Crea las carpetas para las gráficas:**

    Crea las carpetas `confusion_matrices` y `Graficas` con sus subcarpetas (`RegresionLinealMulti`, `RandomForest`, `XGBoost`) y coloca tus archivos PNG generados previamente en las rutas correspondientes, siguiendo la estructura descrita arriba.

6.  **Ejecuta la aplicación Streamlit:**

    ```bash
    streamlit run app.py
    ```
7.  **Explora la aplicación en línea:**

    Puedes disfrutar de la aplicación directamente en línea a través del siguiente enlace: [Valoración Nutricional Antropométrica Multi-Modelo](https://appevaluacion.streamlit.app/)
