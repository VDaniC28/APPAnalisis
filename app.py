import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm # Necesario para el Q-Q plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import warnings
from scipy import stats

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

import scikit_posthocs as sp # Nueva importación

import io
import base64
import os

warnings.filterwarnings('ignore')


# --- Language Configuration ---
LANGUAGES = {
    "es": {
        "page_title": "Evaluación de Modelos ML - Valoración Nutricional",
        "app_title": "🧬 Evaluación de Modelos ML para Valoración Nutricional Antropométrica",
        "app_description": "---",
        "sidebar_title": "⚙️ Configuración",
        "select_language_label": "Selecciona Idioma:",
        "data_load_section": "Carga de Datos",
        "data_loaded_success": "✅ Datos cargados correctamente",
        "records_label": "📊 Registros:",
        "columns_label": "📋 Columnas:",
        "file_not_found_error": "No se encontró el archivo 'datos_pacientes2.csv' en la carpeta actual.",
        "file_load_error": "Error al cargar el archivo: {}",
        "dataset_info_section": "📋 Información del Dataset",
        "first_rows_label": "**Primeras 5 filas:**",
        "statistical_info_label": "**Información estadística:**",
        "target_variables_section": "🎯 Evaluación de Variables Objetivo",
        "start_evaluation_button": "🚀 Iniciar Evaluación de Modelos",
        "preprocessing_spinner": "Preprocesando datos...",
        "column_not_found_error": "La columna '{}' no se encontró en los datos procesados.",
        "data_split_section": "🗂️ División de Datos para {}",
        "train_percent": "**Porcentaje de datos de entrenamiento:** {}%",
        "test_percent": "**Porcentaje de datos de prueba:** {}%",
        "model_training_section": "🤖 Entrenamiento de Modelos para {}",
        "training_spinner": "Entrenando modelos para {}...",
        "training_complete_success": "✅ Modelos entrenados exitosamente para {}",
        "metrics_section": "📊 Métricas de Rendimiento",
        "training_times_section": "⏱️ Tiempos de Entrenamiento",
        "model_label": "Modelo",
        "time_seconds_label": "Tiempo (segundos)",
        "statistical_tests_section": "📈 Pruebas Estadísticas Inferenciales",
        "statistical_tests_spinner": "Realizando pruebas estadísticas para {}...",
        "residuals_normality_test": "Test de Normalidad de Residuos:",
        "shapiro_wilk_test": "{} (Shapiro-Wilk): p-value = {:.4f} ({})",
        "kolmogorov_smirnov_test": "{} (Kolmogorov-Smirnov): p-value = {:.4f} ({})",
        "normal_interpretation": "✅ Normal",
        "not_normal_interpretation": "❌ No Normal",
        "shapiro_wilk_error": "{}: Shapiro-Wilk no pudo ejecutarse: {}",
        "zero_std_error": "{}: Residuos con desviación estándar cero (todos los valores son iguales), no se puede realizar test de normalidad.",
        "no_statistical_results": "No se encontraron resultados de pruebas estadísticas para {}.",
        "evaluation_complete_success": "🎉 Evaluación de modelos completada para todas las variables objetivo.",
        "download_reports_section": "📄 Descargar Reportes PDF",
        "download_pdf_link": "📥 Descargar Reporte PDF para {}",
        "pdf_generation_error": "Error al generar o descargar el reporte PDF para {}: {}",
        "dataset_load_error": "No se pudo cargar el dataset. Verifica que el archivo 'datos_pacientes.csv' esté en la carpeta correcta.",
        "training_linear_regression": "🔄 Entrenando Regresión Lineal...",
        "training_random_forest": "🔄 Entrenando Random Forest...",
        "training_xgboost": "🔄 Entrenando XGBoost...",
        "processed_columns": "Columnas procesadas:",
        "residuals_histogram_title": "Histograma de Residuos para {}",
        "qq_plot_title": "Gráfico Q-Q de Residuos para {}",
        "residuals_vs_predictions_title": "Residuos vs. Predicciones para {}",
        "residuals_label": "Residuos",
        "frequency_label": "Frecuencia",
        "predicted_values_label": "Valores Predichos",
        "friedman_test_heading": "Prueba de Friedman",
        "friedman_result": "Resultado de la prueba de Friedman: Chi-cuadrado = {:.4f}, p-valor = {:.4f} ({})",
        "friedman_significant": "Significativo",
        "friedman_not_significant_interpret": "No Significativo",
        "friedman_not_enough_models": "Se requieren al menos 3 modelos para ejecutar la prueba de Friedman.",
        "friedman_data_error": "Error al preparar los datos para Friedman: {}",
        "friedman_error": "Error al ejecutar la prueba de Friedman: {}",
        "friedman_not_significant": "La prueba de Friedman no fue significativa, no se realizan pruebas post-hoc.",
        "posthoc_heading": "Pruebas Post-Hoc (Nemenyi)",
        "nemenyi_intro": "Resultados de la prueba post-hoc de Nemenyi (valores p):",
        "pdf_friedman_test_heading": "PRUEBA DE FRIEDMAN",
        "pdf_friedman_result": "Estadístico Chi-cuadrado = {:.4f}, p-valor = {:.4f} ({})",
        "pdf_posthoc_heading": "PRUEBAS POST-HOC (NEMENYI)",
        "pdf_nemenyi_intro": "Resultados de la prueba post-hoc de Nemenyi (valores p):",
        "pdf_no_friedman_results": "La prueba de Friedman no pudo ser ejecutada.",
        "pdf_no_posthoc_results": "No se encontraron resultados de pruebas post-hoc (Friedman no fue significativo o hubo un error).",

        # PDF Strings
        "pdf_report_title": "REPORTE DE EVALUACIÓN DE MODELOS ML - {}",
        "pdf_report_subtitle": "Valoración Nutricional Antropométrica",
        "pdf_equipment_heading": "CARACTERÍSTICAS DEL EQUIPO DE PROCESAMIENTO",
        "pdf_component_header": "Componente",
        "pdf_specification_header": "Especificación",
        "pdf_processor": "Procesador",
        "pdf_ram": "RAM instalada",
        "pdf_storage": "Almacenamiento",
        "pdf_gpu": "Tarjeta gráfica",
        "pdf_dataset_info_heading": "INFORMACIÓN DEL DATASET",
        "pdf_num_records": "Número de registros: {}",
        "pdf_num_features": "Número de características: {}",
        "pdf_train_percent": "Porcentaje de entrenamiento: {:.2f}%",
        "pdf_test_percent": "Porcentaje de prueba: {:.2f}%",
        "pdf_training_times_heading": "TIEMPOS DE ENTRENAMIENTO",
        "pdf_metrics_heading": "MÉTRICAS DE RENDIMIENTO",
        "pdf_model_header": "Modelo",
        "pdf_time_seconds_header": "Tiempo (segundos)",
        "pdf_mse": "MSE",
        "pdf_rmse": "RMSE",
        "pdf_mae": "MAE",
        "pdf_r2": "R²",
        "pdf_statistical_tests_heading": "PRUEBAS ESTADÍSTICAS INFERENCIALES",
        "pdf_residuals_normality_heading": "Test de Normalidad de Residuos:",
        "pdf_shapiro_wilk_result": "{} (Shapiro-Wilk): p-value = {:.4f} ({})",
        "pdf_kolmogorov_smirnov_result": "{} (Kolmogorov-Smirnov): p-value = {:.4f} ({})",
        "pdf_shapiro_wilk_note": "{}: Shapiro-Wilk no pudo ejecutarse: {}",
        "pdf_zero_std_note": "{}: Residuos con desviación estándar cero (todos los valores son iguales), no se puede realizar test de normalidad.",
        "pdf_no_stats_found": "No se encontraron resultados de pruebas estadísticas para {}.",
        "pdf_additional_visualizations": "VISUALIZACIONES ADICIONALES",
        "pdf_confusion_matrices": "Matrices de Confusión",
        "pdf_confusion_matrix_for": "Matriz de Confusión para {}:",
        "pdf_confusion_matrix_warning": "Advertencia: Matriz de Confusión para {} no encontrada en {}",
        "pdf_performance_graphs": "Gráficos de Rendimiento por Modelo",
        "pdf_graphs_for_model": "Gráficos para {} ({}):",
        "pdf_graph_title_prefix": "- {}:",
        "pdf_graph_warning": "Advertencia: Gráfico '{}' para {} no encontrado en {}",
        "pdf_target_suffix_warning": "Advertencia: No se encontró sufijo para el objetivo '{}'. No se agregarán gráficos específicos.",
        "pdf_residuals_graphs_heading": "GRÁFICOS DE RESIDUOS"

    },
    "en": {
        "page_title": "ML Model Evaluation - Nutritional Assessment",
        "app_title": "🧬 ML Model Evaluation for Anthropometric Nutritional Assessment",
        "app_description": "---",
        "sidebar_title": "⚙️ Configuration",
        "select_language_label": "Select Language:",
        "data_load_section": "Data Loading",
        "data_loaded_success": "✅ Data loaded successfully",
        "records_label": "📊 Records:",
        "columns_label": "📋 Columns:",
        "file_not_found_error": "File 'datos_pacientes2.csv' not found in the current folder.",
        "file_load_error": "Error loading file: {}",
        "dataset_info_section": "📋 Dataset Information",
        "first_rows_label": "**First 5 Rows:**",
        "statistical_info_label": "**Statistical Information:**",
        "target_variables_section": "🎯 Target Variable Evaluation",
        "start_evaluation_button": "🚀 Start Model Evaluation",
        "preprocessing_spinner": "Preprocessing data...",
        "column_not_found_error": "Column '{}' not found in processed data.",
        "data_split_section": "🗂️ Data Split for {}",
        "train_percent": "**Training data percentage:** {}%",
        "test_percent": "**Test data percentage:** {}%",
        "model_training_section": "🤖 Model Training for {}",
        "training_spinner": "Training models for {}...",
        "training_complete_success": "✅ Models trained successfully for {}",
        "metrics_section": "📊 Performance Metrics",
        "training_times_section": "⏱️ Training Times",
        "model_label": "Model",
        "time_seconds_label": "Time (seconds)",
        "statistical_tests_section": "📈 Inferential Statistical Tests",
        "statistical_tests_spinner": "Performing statistical tests for {}...",
        "residuals_normality_test": "Residuals Normality Test:",
        "shapiro_wilk_test": "{} (Shapiro-Wilk): p-value = {:.4f} ({})",
        "kolmogorov_smirnov_test": "{} (Kolmogorov-Smirnov): p-value = {:.4f} ({})",
        "normal_interpretation": "✅ Normal",
        "not_normal_interpretation": "❌ Not Normal",
        "shapiro_wilk_error": "{}: Shapiro-Wilk could not be performed: {}",
        "zero_std_error": "{}: Residuals with zero standard deviation (all values are the same), normality test cannot be performed.",
        "no_statistical_results": "No statistical test results found for {}.",
        "evaluation_complete_success": "🎉 Model evaluation completed for all target variables.",
        "download_reports_section": "📄 Download PDF Reports",
        "download_pdf_link": "📥 Download PDF Report for {}",
        "pdf_generation_error": "Error generating or downloading PDF report for {}: {}",
        "dataset_load_error": "Could not load the dataset. Please ensure 'datos_pacientes.csv' is in the correct folder.",
        "training_linear_regression": "🔄 Training Linear Regression...",
        "training_random_forest": "🔄 Training Random Forest...",
        "training_xgboost": "🔄 Training XGBoost...",
        "processed_columns": "Processed columns:",
        "residuals_histogram_title": "Residuals Histogram for {}",
        "qq_plot_title": "Residuals Q-Q Plot for {}",
        "residuals_vs_predictions_title": "Residuals vs. Predictions for {}",
        "residuals_label": "Residuals",
        "frequency_label": "Frequency",
        "predicted_values_label": "Predicted Values",
        "friedman_test_heading": "Friedman Test",
        "friedman_result": "Friedman test result: Chi-squared = {:.4f}, p-value = {:.4f} ({})",
        "friedman_significant": "Significant",
        "friedman_not_significant_interpret": "Not Significant",
        "friedman_not_enough_models": "At least 3 models are required to run the Friedman test.",
        "friedman_data_error": "Error preparing data for Friedman: {}",
        "friedman_error": "Error running Friedman test: {}",
        "friedman_not_significant": "Friedman test was not significant, no post-hoc tests performed.",
        "posthoc_heading": "Post-Hoc Tests (Nemenyi)",
        "nemenyi_intro": "Nemenyi post-hoc test results (p-values):",
        "pdf_friedman_test_heading": "FRIEDMAN TEST",
        "pdf_friedman_result": "Chi-squared Statistic = {:.4f}, p-value = {:.4f} ({})",
        "pdf_posthoc_heading": "POST-HOC TESTS (NEMENYI)",
        "pdf_nemenyi_intro": "Nemenyi post-hoc test results (p-values):",
        "pdf_no_friedman_results": "Friedman test could not be executed.",
        "pdf_no_posthoc_results": "No post-hoc test results found (Friedman was not significant or an error occurred).",

        # PDF Strings
        "pdf_report_title": "ML MODEL EVALUATION REPORT - {}",
        "pdf_report_subtitle": "Anthropometric Nutritional Assessment",
        "pdf_equipment_heading": "PROCESSING EQUIPMENT CHARACTERISTICS",
        "pdf_component_header": "Component",
        "pdf_specification_header": "Specification",
        "pdf_processor": "Processor",
        "pdf_ram": "Installed RAM",
        "pdf_storage": "Storage",
        "pdf_gpu": "Graphics Card",
        "pdf_dataset_info_heading": "DATASET INFORMATION",
        "pdf_num_records": "Number of records: {}",
        "pdf_num_features": "Number of features: {}",
        "pdf_train_percent": "Training percentage: {:.2f}%",
        "pdf_test_percent": "Test percentage: {:.2f}%",
        "pdf_training_times_heading": "TRAINING TIMES",
        "pdf_metrics_heading": "PERFORMANCE METRICS",
        "pdf_model_header": "Model",
        "pdf_time_seconds_header": "Time (seconds)",
        "pdf_mse": "MSE",
        "pdf_rmse": "RMSE",
        "pdf_mae": "MAE",
        "pdf_r2": "R²",
        "pdf_statistical_tests_heading": "INFERENTIAL STATISTICAL TESTS",
        "pdf_residuals_normality_heading": "Residuals Normality Test:",
        "pdf_shapiro_wilk_result": "{} (Shapiro-Wilk): p-value = {:.4f} ({})",
        "pdf_kolmogorov_smirnov_result": "{} (Kolmogorov-Smirnov): p-value = {:.4f} ({})",
        "pdf_shapiro_wilk_note": "{}: Shapiro-Wilk could not be performed: {}",
        "pdf_zero_std_note": "{}: Residuals with zero standard deviation (all values are the same), normality test cannot be performed.",
        "pdf_no_stats_found": "No statistical test results found for {}.",
        "pdf_additional_visualizations": "ADDITIONAL VISUALIZATIONS",
        "pdf_confusion_matrices": "Confusion Matrices",
        "pdf_confusion_matrix_for": "Confusion Matrix for {}:",
        "pdf_confusion_matrix_warning": "Warning: Confusion Matrix for {} not found at {}",
        "pdf_performance_graphs": "Model Performance Graphs",
        "pdf_graphs_for_model": "Graphs for {} ({}):",
        "pdf_graph_title_prefix": "- {}:",
        "pdf_graph_warning": "Warning: Graph '{}' for {} not found at {}",
        "pdf_target_suffix_warning": "Warning: No suffix found for target '{}'. Specific graphs will not be added.",
        "pdf_residuals_graphs_heading": "RESIDUALS GRAPHS"

    },

    "zh": { # Chinese (Simplified)
        "page_title": "机器学习模型评估 - 营养评估",
        "app_title": "🧬 人体测量营养评估的机器学习模型评估",
        "app_description": "---",
        "sidebar_title": "⚙️ 配置",
        "select_language_label": "选择语言:",
        "data_load_section": "数据加载",
        "data_loaded_success": "✅ 数据加载成功",
        "records_label": "📊 记录:",
        "columns_label": "📋 列:",
        "file_not_found_error": "在当前文件夹中找不到文件 'datos_pacientes2.csv'。",
        "file_load_error": "加载文件时出错: {}",
        "dataset_info_section": "📋 数据集信息",
        "first_rows_label": "**前5行:**",
        "statistical_info_label": "**统计信息:**",
        "target_variables_section": "🎯 目标变量评估",
        "start_evaluation_button": "🚀 开始模型评估",
        "preprocessing_spinner": "正在预处理数据...",
        "column_not_found_error": "在已处理数据中找不到列 '{}'。",
        "data_split_section": "🗂️ {} 的数据分割",
        "train_percent": "**训练数据百分比:** {}%",
        "test_percent": "**测试数据百分比:** {}%",
        "model_training_section": "🤖 {} 的模型训练",
        "training_spinner": "正在为 {} 训练模型...",
        "training_complete_success": "✅ 已成功为 {} 训练模型",
        "metrics_section": "📊 性能指标",
        "training_times_section": "⏱️ 训练时间",
        "model_label": "模型",
        "time_seconds_label": "时间 (秒)",
        "statistical_tests_section": "📈 推断统计检验",
        "statistical_tests_spinner": "正在为 {} 执行统计检验...",
        "residuals_normality_test": "残差正态性检验:",
        "shapiro_wilk_test": "{} (Shapiro-Wilk): p值 = {:.4f} ({})",
        "kolmogorov_smirnov_test": "{} (Kolmogorov-Smirnov): p值 = {:.4f} ({})",
        "normal_interpretation": "✅ 正常",
        "not_normal_interpretation": "❌ 不正常",
        "shapiro_wilk_error": "{}: Shapiro-Wilk 无法执行: {}",
        "zero_std_error": "{}: 残差标准差为零（所有值都相同），无法执行正态性检验。",
        "no_statistical_results": "未找到 {} 的统计检验结果。",
        "evaluation_complete_success": "🎉 所有目标变量的模型评估已完成。",
        "download_reports_section": "📄 下载 PDF 报告",
        "download_pdf_link": "📥 下载 {} 的 PDF 报告",
        "pdf_generation_error": "为 {} 生成或下载 PDF 报告时出错: {}",
        "dataset_load_error": "无法加载数据集。请确保 'datos_pacientes.csv' 在正确的文件夹中。",
        "training_linear_regression": "🔄 正在训练线性回归...",
        "training_random_forest": "🔄 正在训练随机森林...",
        "training_xgboost": "🔄 正在训练 XGBoost...",
        "processed_columns": "已处理的列:",
        "residuals_histogram_title": "{} 的残差直方图",
        "qq_plot_title": "{} 的残差 Q-Q 图",
        "residuals_vs_predictions_title": "{} 的残差与预测值",
        "residuals_label": "残差",
        "frequency_label": "频率",
        "predicted_values_label": "预测值",
         "friedman_test_heading": "Friedman 检验",
        "friedman_result": "Friedman 检验结果: 卡方 = {:.4f}, p 值 = {:.4f} ({})",
        "friedman_significant": "显著",
        "friedman_not_significant_interpret": "不显著",
        "friedman_not_enough_models": "运行 Friedman 检验至少需要 3 个模型。",
        "friedman_data_error": "准备 Friedman 数据时出错: {}",
        "friedman_error": "运行 Friedman 检验时出错: {}",
        "friedman_not_significant": "Friedman 检验不显著，未执行事后检验。",
        "posthoc_heading": "事后检验 (Nemenyi)",
        "nemenyi_intro": "Nemenyi 事后检验结果 (p 值):",
        "pdf_friedman_test_heading": "FRIEDMAN 检验",
        "pdf_friedman_result": "卡方统计量 = {:.4f}, p 值 = {:.4f} ({})",
        "pdf_posthoc_heading": "事后检验 (NEMENYI)",
        "pdf_nemenyi_intro": "Nemenyi 事后检验结果 (p 值):",
        "pdf_no_friedman_results": "Friedman 检验无法执行。",
        "pdf_no_posthoc_results": "未找到事后检验结果（Friedman 不显著或发生错误）。",

        # PDF Strings
        "pdf_report_title": "ML 模型评估报告 - {}",
        "pdf_report_subtitle": "人体测量营养评估",
        "pdf_equipment_heading": "处理设备特性",
        "pdf_component_header": "组件",
        "pdf_specification_header": "规格",
        "pdf_processor": "处理器",
        "pdf_ram": "已安装内存",
        "pdf_storage": "存储",
        "pdf_gpu": "显卡",
        "pdf_dataset_info_heading": "数据集信息",
        "pdf_num_records": "记录数: {}",
        "pdf_num_features": "特征数: {}",
        "pdf_train_percent": "训练百分比: {:.2f}%",
        "pdf_test_percent": "测试百分比: {:.2f}%",
        "pdf_training_times_heading": "训练时间",
        "pdf_metrics_heading": "性能指标",
        "pdf_model_header": "模型",
        "pdf_time_seconds_header": "时间 (秒)",
        "pdf_mse": "均方误差",
        "pdf_rmse": "均方根误差",
        "pdf_mae": "平均绝对误差",
        "pdf_r2": "决定系数 (R²)",
        "pdf_statistical_tests_heading": "推断统计检验",
        "pdf_residuals_normality_heading": "残差正态性检验:",
        "pdf_shapiro_wilk_result": "{} (Shapiro-Wilk): p值 = {:.4f} ({})",
        "pdf_kolmogorov_smirnov_result": "{} (Kolmogorov-Smirnov): p值 = {:.4f} ({})",
        "pdf_shapiro_wilk_note": "{}: Shapiro-Wilk 无法执行: {}",
        "pdf_zero_std_note": "{}: 残差标准差为零（所有值都相同），无法执行正态性检验。",
        "pdf_no_stats_found": "未找到 {} 的统计检验结果。",
        "pdf_additional_visualizations": "附加可视化",
        "pdf_confusion_matrices": "混淆矩阵",
        "pdf_confusion_matrix_for": "{} 的混淆矩阵:",
        "pdf_confusion_matrix_warning": "警告: 在 {} 未找到 {} 的混淆矩阵",
        "pdf_performance_graphs": "模型性能图",
        "pdf_graphs_for_model": "{} ({}) 的图:",
        "pdf_graph_title_prefix": "- {}:",
        "pdf_graph_warning": "警告: 在 {} 未找到 {} 的图 '{}'",
        "pdf_target_suffix_warning": "警告: 未找到目标 '{}' 的后缀。将不添加特定图。",
        "pdf_residuals_graphs_heading": "残差图"
    },
    "de": { # German
        "page_title": "ML-Modellbewertung - Ernährungsanalyse",
        "app_title": "🧬 ML-Modellbewertung für anthropometrische Ernährungsanalyse",
        "app_description": "---",
        "sidebar_title": "⚙️ Konfiguration",
        "select_language_label": "Sprache auswählen:",
        "data_load_section": "Daten laden",
        "data_loaded_success": "✅ Daten erfolgreich geladen",
        "records_label": "📊 Datensätze:",
        "columns_label": "📋 Spalten:",
        "file_not_found_error": "Datei 'datos_pacientes2.csv' im aktuellen Ordner nicht gefunden.",
        "file_load_error": "Fehler beim Laden der Datei: {}",
        "dataset_info_section": "📋 Datensatzinformationen",
        "first_rows_label": "**Erste 5 Zeilen:**",
        "statistical_info_label": "**Statistische Informationen:**",
        "target_variables_section": "🎯 Zielvariablenbewertung",
        "start_evaluation_button": "🚀 Modellbewertung starten",
        "preprocessing_spinner": "Daten werden vorverarbeitet...",
        "column_not_found_error": "Spalte '{}' in den verarbeiteten Daten nicht gefunden.",
        "data_split_section": "🗂️ Datenteilung für {}",
        "train_percent": "**Prozentsatz der Trainingsdaten:** {}%",
        "test_percent": "**Prozentsatz der Testdaten:** {}%",
        "model_training_section": "🤖 Modelltraining für {}",
        "training_spinner": "Modelle werden für {} trainiert...",
        "training_complete_success": "✅ Modelle erfolgreich für {} trainiert",
        "metrics_section": "📊 Leistungsmetriken",
        "training_times_section": "⏱️ Trainingszeiten",
        "model_label": "Modell",
        "time_seconds_label": "Zeit (Sekunden)",
        "statistical_tests_section": "📈 Inferenzstatistische Tests",
        "statistical_tests_spinner": "Statistische Tests für {} werden durchgeführt...",
        "residuals_normality_test": "Normalitätstest der Residuen:",
        "shapiro_wilk_test": "{} (Shapiro-Wilk): p-Wert = {:.4f} ({})",
        "kolmogorov_smirnov_test": "{} (Kolmogorov-Smirnov): p-Wert = {:.4f} ({})",
        "normal_interpretation": "✅ Normal",
        "not_normal_interpretation": "❌ Nicht Normal",
        "shapiro_wilk_error": "{}: Shapiro-Wilk konnte nicht durchgeführt werden: {}",
        "zero_std_error": "{}: Residuen mit Standardabweichung Null (alle Werte sind gleich), Normalitätstest kann nicht durchgeführt werden.",
        "no_statistical_results": "Keine statistischen Testergebnisse für {} gefunden.",
        "evaluation_complete_success": "🎉 Modellbewertung für alle Zielvariablen abgeschlossen.",
        "download_reports_section": "📄 PDF-Berichte herunterladen",
        "download_pdf_link": "📥 PDF-Bericht für {} herunterladen",
        "pdf_generation_error": "Fehler beim Generieren oder Herunterladen des PDF-Berichts für {}: {}",
        "dataset_load_error": "Datensatz konnte nicht geladen werden. Stellen Sie sicher, dass 'datos_pacientes.csv' im richtigen Ordner ist.",
        "training_linear_regression": "🔄 Lineare Regression wird trainiert...",
        "training_random_forest": "🔄 Random Forest wird trainiert...",
        "training_xgboost": "🔄 XGBoost wird trainiert...",
        "processed_columns": "Verarbeitete Spalten:",
        "residuals_histogram_title": "Residuen-Histogramm für {}",
        "qq_plot_title": "Residuen-Q-Q-Diagramm für {}",
        "residuals_vs_predictions_title": "Residuen vs. Vorhersagen für {}",
        "residuals_label": "Residuen",
        "frequency_label": "Häufigkeit",
        "predicted_values_label": "Vorhergesagte Werte",
        "friedman_test_heading": "Friedman-Test",
        "friedman_result": "Friedman-Testergebnis: Chi-Quadrat = {:.4f}, p-Wert = {:.4f} ({})",
        "friedman_significant": "Signifikant",
        "friedman_not_significant_interpret": "Nicht signifikant",
        "friedman_not_enough_models": "Es werden mindestens 3 Modelle für den Friedman-Test benötigt.",
        "friedman_data_error": "Fehler beim Vorbereiten der Daten für Friedman: {}",
        "friedman_error": "Fehler beim Ausführen des Friedman-Tests: {}",
        "friedman_not_significant": "Friedman-Test war nicht signifikant, keine Post-hoc-Tests durchgeführt.",
        "posthoc_heading": "Post-hoc-Tests (Nemenyi)",
        "nemenyi_intro": "Nemenyi Post-hoc-Testergebnisse (p-Werte):",
        "pdf_friedman_test_heading": "FRIEDMAN-TEST",
        "pdf_friedman_result": "Chi-Quadrat-Statistik = {:.4f}, p-Wert = {:.4f} ({})",
        "pdf_posthoc_heading": "POST-HOC-TESTS (NEMENYI)",
        "pdf_nemenyi_intro": "Nemenyi Post-hoc-Testergebnisse (p-Werte):",
        "pdf_no_friedman_results": "Friedman-Test konnte nicht ausgeführt werden.",
        "pdf_no_posthoc_results": "Keine Post-hoc-Testergebnisse gefunden (Friedman war nicht signifikant oder es ist ein Fehler aufgetreten).",

        # PDF Strings
        "pdf_report_title": "ML-MODELLBEWERTUNGSBERICHT - {}",
        "pdf_report_subtitle": "Anthropometrische Ernährungsanalyse",
        "pdf_equipment_heading": "EIGENSCHAFTEN DER VERARBEITUNGSAUSRÜSTUNG",
        "pdf_component_header": "Komponente",
        "pdf_specification_header": "Spezifikation",
        "pdf_processor": "Prozessor",
        "pdf_ram": "Installierter RAM",
        "pdf_storage": "Speicher",
        "pdf_gpu": "Grafikkarte",
        "pdf_dataset_info_heading": "DATENSATZINFORMATIONEN",
        "pdf_num_records": "Anzahl der Datensätze: {}",
        "pdf_num_features": "Anzahl der Merkmale: {}",
        "pdf_train_percent": "Trainingsprozentsatz: {:.2f}%",
        "pdf_test_percent": "Testprozentsatz: {:.2f}%",
        "pdf_training_times_heading": "TRAININGSZEITEN",
        "pdf_metrics_heading": "LEISTUNGSMETRIKEN",
        "pdf_model_header": "Modell",
        "pdf_time_seconds_header": "Zeit (Sekunden)",
        "pdf_mse": "MSE",
        "pdf_rmse": "RMSE",
        "pdf_mae": "MAE",
        "pdf_r2": "R²",
        "pdf_statistical_tests_heading": "INFERENZSTATISTISCHE TESTS",
        "pdf_residuals_normality_heading": "Normalitätstest der Residuen:",
        "pdf_shapiro_wilk_result": "{} (Shapiro-Wilk): p-Wert = {:.4f} ({})",
        "pdf_kolmogorov_smirnov_result": "{} (Kolmogorov-Smirnov): p-Wert = {:.4f} ({})",
        "pdf_shapiro_wilk_note": "{}: Shapiro-Wilk konnte nicht durchgeführt werden: {}",
        "pdf_zero_std_note": "{}: Residuen mit Standardabweichung Null (alle Werte sind gleich), Normalitätstest kann nicht durchgeführt werden.",
        "pdf_no_stats_found": "Keine statistischen Testergebnisse für {} gefunden.",
        "pdf_additional_visualizations": "ZUSÄTZLICHE VISUALISIERUNGEN",
        "pdf_confusion_matrices": "Konfusionsmatrizen",
        "pdf_confusion_matrix_for": "Konfusionsmatrix für {}:",
        "pdf_confusion_matrix_warning": "Warnung: Konfusionsmatrix für {} nicht gefunden unter {}",
        "pdf_performance_graphs": "Modellleistungsdiagramme",
        "pdf_graphs_for_model": "Diagramme für {} ({}):",
        "pdf_graph_title_prefix": "- {}:",
        "pdf_graph_warning": "Warnung: Diagramm '{}' für {} nicht gefunden unter {}",
        "pdf_target_suffix_warning": "Warnung: Kein Suffix für Ziel '{}' gefunden. Spezifische Diagramme werden nicht hinzugefügt.",
        "pdf_residuals_graphs_heading": "RESIDUEN-DIAGRAMME"
    },
    "ja": { # Japanese
        "page_title": "MLモデル評価 - 栄養評価",
        "app_title": "🧬 人体計測栄養評価のためのMLモデル評価",
        "app_description": "---",
        "sidebar_title": "⚙️ 設定",
        "select_language_label": "言語を選択:",
        "data_load_section": "データ読み込み",
        "data_loaded_success": "✅ データの読み込みに成功しました",
        "records_label": "📊 レコード数:",
        "columns_label": "📋 列数:",
        "file_not_found_error": "現在のフォルダにファイル 'datos_pacientes2.csv' が見つかりませんでした。",
        "file_load_error": "ファイルの読み込みエラー: {}",
        "dataset_info_section": "📋 データセット情報",
        "first_rows_label": "**最初の5行:**",
        "statistical_info_label": "**統計情報:**",
        "target_variables_section": "🎯 目標変数評価",
        "start_evaluation_button": "🚀 モデル評価を開始",
        "preprocessing_spinner": "データを前処理中...",
        "column_not_found_error": "処理済みデータに列 '{}' が見つかりませんでした。",
        "data_split_section": "🗂️ {} のデータ分割",
        "train_percent": "**トレーニングデータ割合:** {}%",
        "test_percent": "**テストデータ割合:** {}%",
        "model_training_section": "🤖 {} のモデルトレーニング",
        "training_spinner": "{} のモデルをトレーニング中...",
        "training_complete_success": "✅ {} のモデルが正常にトレーニングされました",
        "metrics_section": "📊 パフォーマンス指標",
        "training_times_section": "⏱️ トレーニング時間",
        "model_label": "モデル",
        "time_seconds_label": "時間 (秒)",
        "statistical_tests_section": "📈 推測統計検定",
        "statistical_tests_spinner": "{} の統計検定を実行中...",
        "residuals_normality_test": "残差の正規性検定:",
        "shapiro_wilk_test": "{} (Shapiro-Wilk): p値 = {:.4f} ({})",
        "kolmogorov_smirnov_test": "{} (Kolmogorov-Smirnov): p値 = {:.4f} ({})",
        "normal_interpretation": "✅ 正規",
        "not_normal_interpretation": "❌ 非正規",
        "shapiro_wilk_error": "{}: Shapiro-Wilk を実行できませんでした: {}",
        "zero_std_error": "{}: 残差の標準偏差がゼロ（すべての値が同じ）のため、正規性検定は実行できません。",
        "no_statistical_results": "{} の統計検定結果が見つかりませんでした。",
        "evaluation_complete_success": "🎉 すべての目標変数のモデル評価が完了しました。",
        "download_reports_section": "📄 PDFレポートをダウンロード",
        "download_pdf_link": "📥 {} のPDFレポートをダウンロード",
        "pdf_generation_error": "{} のPDFレポートの生成またはダウンロード中にエラーが発生しました: {}",
        "dataset_load_error": "データセットをロードできませんでした。'datos_pacientes.csv' が正しいフォルダにあることを確認してください。",
        "training_linear_regression": "🔄 線形回帰をトレーニング中...",
        "training_random_forest": "🔄 ランダムフォレストをトレーニング中...",
        "training_xgboost": "🔄 XGBoostをトレーニング中...",
        "processed_columns": "処理済み列:",
        "residuals_histogram_title": "{} の残差ヒストグラム",
        "qq_plot_title": "{} の残差 Q-Q プロット",
        "residuals_vs_predictions_title": "{} の残差 vs. 予測",
        "residuals_label": "残差",
        "frequency_label": "頻度",
        "predicted_values_label": "予測値",
        "friedman_test_heading": "フリードマン検定",
        "friedman_result": "フリードマン検定結果: カイ二乗 = {:.4f}, p 値 = {:.4f} ({})",
        "friedman_significant": "有意",
        "friedman_not_significant_interpret": "有意でない",
        "friedman_not_enough_models": "フリードマン検定を実行するには、少なくとも3つのモデルが必要です。",
        "friedman_data_error": "フリードマンのデータを準備中にエラーが発生しました: {}",
        "friedman_error": "フリードマン検定の実行中にエラーが発生しました: {}",
        "friedman_not_significant": "フリードマン検定は有意ではなかったため、事後検定は実行されません。",
        "posthoc_heading": "事後検定 (Nemenyi)",
        "nemenyi_intro": "Nemenyi 事後検定結果 (p 値):",
        "pdf_friedman_test_heading": "フリードマン検定",
        "pdf_friedman_result": "カイ二乗統計量 = {:.4f}, p 値 = {:.4f} ({})",
        "pdf_posthoc_heading": "事後検定 (NEMENYI)",
        "pdf_nemenyi_intro": "Nemenyi 事後検定結果 (p 値):",
        "pdf_no_friedman_results": "フリードマン検定を実行できませんでした。",
        "pdf_no_posthoc_results": "事後検定の結果が見つかりませんでした (フリードマンは有意ではなかったか、エラーが発生しました)。",

        # PDF Strings
        "pdf_report_title": "MLモデル評価レポート - {}",
        "pdf_report_subtitle": "人体計測栄養評価",
        "pdf_equipment_heading": "処理装置の特性",
        "pdf_component_header": "コンポーネント",
        "pdf_specification_header": "仕様",
        "pdf_processor": "プロセッサ",
        "pdf_ram": "インストール済みRAM",
        "pdf_storage": "ストレージ",
        "pdf_gpu": "グラフィックカード",
        "pdf_dataset_info_heading": "データセット情報",
        "pdf_num_records": "レコード数: {}",
        "pdf_num_features": "特徴量数: {}",
        "pdf_train_percent": "トレーニング割合: {:.2f}%",
        "pdf_test_percent": "テスト割合: {:.2f}%",
        "pdf_training_times_heading": "トレーニング時間",
        "pdf_metrics_heading": "パフォーマンス指標",
        "pdf_model_header": "モデル",
        "pdf_time_seconds_header": "時間 (秒)",
        "pdf_mse": "MSE",
        "pdf_rmse": "RMSE",
        "pdf_mae": "MAE",
        "pdf_r2": "R²",
        "pdf_statistical_tests_heading": "推測統計検定",
        "pdf_residuals_normality_heading": "残差の正規性検定:",
        "pdf_shapiro_wilk_result": "{} (Shapiro-Wilk): p値 = {:.4f} ({})",
        "pdf_kolmogorov_smirnov_result": "{} (Kolmogorov-Smirnov): p値 = {:.4f} ({})",
        "pdf_shapiro_wilk_note": "{}: Shapiro-Wilk を実行できませんでした: {}",
        "pdf_zero_std_note": "{}: 残差の標準偏差がゼロ（すべての値が同じ）のため、正規性検定は実行できません。",
        "pdf_no_stats_found": "{} の統計検定結果が見つかりませんでした。",
        "pdf_additional_visualizations": "追加の視覚化",
        "pdf_confusion_matrices": "混同行列",
        "pdf_confusion_matrix_for": "{} の混同行列:",
        "pdf_confusion_matrix_warning": "警告: {} の混同行列が {} に見つかりませんでした",
        "pdf_performance_graphs": "モデル性能グラフ",
        "pdf_graphs_for_model": "{} ({}) のグラフ:",
        "pdf_graph_title_prefix": "- {}:",
        "pdf_graph_warning": "警告: {} のグラフ '{}' が {} に見つかりませんでした",
        "pdf_target_suffix_warning": "警告: ターゲット '{}' のサフィックスが見つかりませんでした。特定のグラフは追加されません。",
        "pdf_residuals_graphs_heading": "残差グラフ"
    },
    "fr": { # French
        "page_title": "Évaluation de Modèles ML - Évaluation Nutritionnelle",
        "app_title": "🧬 Évaluation de Modèles ML pour l'Évaluation Nutritionnelle Anthropométrique",
        "app_description": "---",
        "sidebar_title": "⚙️ Configuration",
        "select_language_label": "Sélectionner la langue :",
        "data_load_section": "Chargement des Données",
        "data_loaded_success": "✅ Données chargées avec succès",
        "records_label": "📊 Enregistrements :",
        "columns_label": "📋 Colonnes :",
        "file_not_found_error": "Fichier 'datos_pacientes2.csv' introuvable dans le dossier actuel.",
        "file_load_error": "Erreur lors du chargement du fichier : {}",
        "dataset_info_section": "📋 Informations sur l'Ensemble de Données",
        "first_rows_label": "**5 premières lignes :**",
        "statistical_info_label": "**Informations statistiques :**",
        "target_variables_section": "🎯 Évaluation des Variables Cibles",
        "start_evaluation_button": "🚀 Démarrer l'Évaluation des Modèles",
        "preprocessing_spinner": "Prétraitement des données...",
        "column_not_found_error": "La colonne '{}' n'a pas été trouvée dans les données traitées.",
        "data_split_section": "🗂️ Division des Données pour {}",
        "train_percent": "**Pourcentage de données d'entraînement :** {}%",
        "test_percent": "**Pourcentage de données de test :** {}%",
        "model_training_section": "🤖 Entraînement des Modèles pour {}",
        "training_spinner": "Entraînement des modèles pour {}...",
        "training_complete_success": "✅ Modèles entraînés avec succès pour {}",
        "metrics_section": "📊 Métriques de Performance",
        "training_times_section": "⏱️ Temps d'Entraînement",
        "model_label": "Modèle",
        "time_seconds_label": "Temps (secondes)",
        "statistical_tests_section": "📈 Tests Statistiques Inférentiels",
        "statistical_tests_spinner": "Exécution des tests statistiques pour {}...",
        "residuals_normality_test": "Test de Normalité des Résidus :",
        "shapiro_wilk_test": "{} (Shapiro-Wilk) : p-value = {:.4f} ({})",
        "kolmogorov_smirnov_test": "{} (Kolmogorov-Smirnov) : p-value = {:.4f} ({})",
        "normal_interpretation": "✅ Normal",
        "not_normal_interpretation": "❌ Non Normal",
        "shapiro_wilk_error": "{}: Shapiro-Wilk n'a pas pu être exécuté : {}",
        "zero_std_error": "{}: Résidus avec écart type nul (toutes les valeurs sont identiques), le test de normalité ne peut pas être effectué.",
        "no_statistical_results": "Aucun résultat de test statistique trouvé pour {}.",
        "evaluation_complete_success": "🎉 Évaluation des modèles terminée pour toutes les variables cibles.",
        "download_reports_section": "📄 Télécharger les Rapports PDF",
        "download_pdf_link": "📥 Télécharger le Rapport PDF pour {}",
        "pdf_generation_error": "Erreur lors de la génération ou du téléchargement du rapport PDF pour {} : {}",
        "dataset_load_error": "Impossible de charger l'ensemble de données. Veuillez vous assurer que 'datos_pacientes.csv' se trouve dans le bon dossier.",
        "training_linear_regression": "🔄 Entraînement de la Régression Linéaire...",
        "training_random_forest": "🔄 Entraînement de Random Forest...",
        "training_xgboost": "🔄 Entraînement de XGBoost...",
        "processed_columns": "Colonnes traitées :",
        "residuals_histogram_title": "Histogramme des Résidus pour {}",
        "qq_plot_title": "Graphe Q-Q des Résidus pour {}",
        "residuals_vs_predictions_title": "Résidus vs. Prédictions pour {}",
        "residuals_label": "Résidus",
        "frequency_label": "Fréquence",
        "predicted_values_label": "Valeurs Prédites",
        "friedman_test_heading": "Test de Friedman",
        "friedman_result": "Résultat du test de Friedman: Chi-deux = {:.4f}, p-valeur = {:.4f} ({})",
        "friedman_significant": "Significatif",
        "friedman_not_significant_interpret": "Non Significatif",
        "friedman_not_enough_models": "Au moins 3 modèles sont requis pour exécuter le test de Friedman.",
        "friedman_data_error": "Erreur lors de la préparation des données pour Friedman: {}",
        "friedman_error": "Erreur lors de l'exécution du test de Friedman: {}",
        "friedman_not_significant": "Le test de Friedman n'était pas significatif, aucun test post-hoc effectué.",
        "posthoc_heading": "Tests Post-Hoc (Nemenyi)",
        "nemenyi_intro": "Résultats du test post-hoc de Nemenyi (valeurs p):",
        "pdf_friedman_test_heading": "TEST DE FRIEDMAN",
        "pdf_friedman_result": "Statistique du Chi-deux = {:.4f}, p-valeur = {:.4f} ({})",
        "pdf_posthoc_heading": "TESTS POST-HOC (NEMENYI)",
        "pdf_nemenyi_intro": "Résultats du test post-hoc de Nemenyi (valeurs p):",
        "pdf_no_friedman_results": "Le test de Friedman n'a pas pu être exécuté.",
        "pdf_no_posthoc_results": "Aucun résultat de test post-hoc trouvé (Friedman n'était pas significatif ou une erreur s'est produite).",

        # PDF Strings
        "pdf_report_title": "RAPPORT D'ÉVALUATION DES MODÈLES ML - {}",
        "pdf_report_subtitle": "Évaluation Nutritionnelle Anthropométrique",
        "pdf_equipment_heading": "CARACTÉRISTIQUES DE L'ÉQUIPEMENT DE TRAITEMENT",
        "pdf_component_header": "Composant",
        "pdf_specification_header": "Spécification",
        "pdf_processor": "Processeur",
        "pdf_ram": "RAM installée",
        "pdf_storage": "Stockage",
        "pdf_gpu": "Carte graphique",
        "pdf_dataset_info_heading": "INFORMATIONS SUR L'ENSEMBLE DE DONNÉES",
        "pdf_num_records": "Nombre d'enregistrements : {}",
        "pdf_num_features": "Nombre de caractéristiques : {}",
        "pdf_train_percent": "Pourcentage d'entraînement : {:.2f}%",
        "pdf_test_percent": "Pourcentage de test : {:.2f}%",
        "pdf_training_times_heading": "TEMPS D'ENTRAÎNEMENT",
        "pdf_metrics_heading": "MÉTRIQUES DE PERFORMANCE",
        "pdf_model_header": "Modèle",
        "pdf_time_seconds_header": "Temps (secondes)",
        "pdf_mse": "MSE",
        "pdf_rmse": "RMSE",
        "pdf_mae": "MAE",
        "pdf_r2": "R²",
        "pdf_statistical_tests_heading": "TESTS STATISTIQUES INFÉRENTIELS",
        "pdf_residuals_normality_heading": "Test de Normalité des Résidus :",
        "pdf_shapiro_wilk_result": "{} (Shapiro-Wilk) : p-value = {:.4f} ({})",
        "pdf_kolmogorov_smirnov_result": "{} (Kolmogorov-Smirnov) : p-value = {:.4f} ({})",
        "pdf_shapiro_wilk_note": "{}: Shapiro-Wilk n'a pas pu être exécuté : {}",
        "pdf_zero_std_note": "{}: Résidus avec écart type nul (toutes les valeurs sont identiques), le test de normalité ne peut pas être effectué.",
        "pdf_no_stats_found": "Aucun résultat de test statistique trouvé pour {}.",
        "pdf_additional_visualizations": "VISUALISATIONS SUPPLÉMENTAIRES",
        "pdf_confusion_matrices": "Matrices de Confusion",
        "pdf_confusion_matrix_for": "Matrice de Confusion pour {} :",
        "pdf_confusion_matrix_warning": "Avertissement : Matrice de Confusion pour {} introuvable à {}",
        "pdf_performance_graphs": "Graphiques de Performance du Modèle",
        "pdf_graphs_for_model": "Graphiques pour {} ({}):",
        "pdf_graph_title_prefix": "- {}:",
        "pdf_graph_warning": "Avertissement : Graphique '{}' pour {} introuvable à {}",
        "pdf_target_suffix_warning": "Avertissement : Aucun suffixe trouvé pour la cible '{}'. Les graphiques spécifiques ne seront pas ajoutés.",
        "pdf_residuals_graphs_heading": "GRAPHIQUES DES RÉSIDUS"
    },
    "pt": { # Portuguese (Brazil)
        "page_title": "Avaliação de Modelos ML - Avaliação Nutricional",
        "app_title": "🧬 Avaliação de Modelos ML para Avaliação Nutricional Antropométrica",
        "app_description": "---",
        "sidebar_title": "⚙️ Configuração",
        "select_language_label": "Selecionar Idioma:",
        "data_load_section": "Carregamento de Dados",
        "data_loaded_success": "✅ Dados carregados com sucesso",
        "records_label": "📊 Registros:",
        "columns_label": "📋 Colunas:",
        "file_not_found_error": "Arquivo 'datos_pacientes2.csv' não encontrado na pasta atual.",
        "file_load_error": "Erro ao carregar o arquivo: {}",
        "dataset_info_section": "📋 Informações do Conjunto de Dados",
        "first_rows_label": "**Primeiras 5 linhas:**",
        "statistical_info_label": "**Informações estatísticas:**",
        "target_variables_section": "🎯 Avaliação de Variáveis Alvo",
        "start_evaluation_button": "🚀 Iniciar Avaliação de Modelos",
        "preprocessing_spinner": "Pré-processando dados...",
        "column_not_found_error": "A coluna '{}' não foi encontrada nos dados processados.",
        "data_split_section": "🗂️ Divisão de Dados para {}",
        "train_percent": "**Porcentagem de dados de treinamento:** {}%",
        "test_percent": "**Porcentagem de dados de teste:** {}%",
        "model_training_section": "🤖 Treinamento de Modelos para {}",
        "training_spinner": "Treinando modelos para {}...",
        "training_complete_success": "✅ Modelos treinados com sucesso para {}",
        "metrics_section": "📊 Métricas de Desempenho",
        "training_times_section": "⏱️ Tempos de Treinamento",
        "model_label": "Modelo",
        "time_seconds_label": "Tempo (segundos)",
        "statistical_tests_section": "📈 Testes Estatísticos Inferenciais",
        "statistical_tests_spinner": "Realizando testes estatísticos para {}...",
        "residuals_normality_test": "Teste de Normalidade dos Resíduos:",
        "shapiro_wilk_test": "{} (Shapiro-Wilk): p-valor = {:.4f} ({})",
        "kolmogorov_smirnov_test": "{} (Kolmogorov-Smirnov): p-valor = {:.4f} ({})",
        "normal_interpretation": "✅ Normal",
        "not_normal_interpretation": "❌ Não Normal",
        "shapiro_wilk_error": "{}: Shapiro-Wilk não pôde ser executado: {}",
        "zero_std_error": "{}: Resíduos com desvio padrão zero (todos os valores são iguais), o teste de normalidade não pode ser realizado.",
        "no_statistical_results": "Nenhum resultado de teste estatístico encontrado para {}.",
        "evaluation_complete_success": "🎉 Avaliação de modelos concluída para todas as variáveis alvo.",
        "download_reports_section": "📄 Baixar Relatórios PDF",
        "download_pdf_link": "📥 Baixar Relatório PDF para {}",
        "pdf_generation_error": "Erro ao gerar ou baixar o relatório PDF para {}: {}",
        "dataset_load_error": "Não foi possível carregar o conjunto de dados. Verifique se o arquivo 'datos_pacientes.csv' está na pasta correta.",
        "training_linear_regression": "🔄 Treinando Regressão Linear...",
        "training_random_forest": "🔄 Treinando Random Forest...",
        "training_xgboost": "🔄 Treinando XGBoost...",
        "processed_columns": "Colunas processadas:",
        "residuals_histogram_title": "Histograma de Resíduos para {}",
        "qq_plot_title": "Gráfico Q-Q de Resíduos para {}",
        "residuals_vs_predictions_title": "Resíduos vs. Previsões para {}",
        "residuals_label": "Resíduos",
        "frequency_label": "Frequência",
        "predicted_values_label": "Valores Previstos",
        "friedman_test_heading": "Teste de Friedman",
        "friedman_result": "Resultado do teste de Friedman: Qui-quadrado = {:.4f}, p-valor = {:.4f} ({})",
        "friedman_significant": "Significativo",
        "friedman_not_significant_interpret": "Não Significativo",
        "friedman_not_enough_models": "São necessários pelo menos 3 modelos para executar o teste de Friedman.",
        "friedman_data_error": "Erro ao preparar os dados para Friedman: {}",
        "friedman_error": "Erro ao executar o teste de Friedman: {}",
        "friedman_not_significant": "O teste de Friedman não foi significativo, nenhum teste post-hoc realizado.",
        "posthoc_heading": "Testes Post-Hoc (Nemenyi)",
        "nemenyi_intro": "Resultados do teste post-hoc de Nemenyi (valores p):",
        "pdf_friedman_test_heading": "TESTE DE FRIEDMAN",
        "pdf_friedman_result": "Estatística Qui-quadrado = {:.4f}, p-valor = {:.4f} ({})",
        "pdf_posthoc_heading": "TESTES POST-HOC (NEMENYI)",
        "pdf_nemenyi_intro": "Resultados do teste post-hoc de Nemenyi (valores p):",
        "pdf_no_friedman_results": "O teste de Friedman não pôde ser executado.",
        "pdf_no_posthoc_results": "Nenhum resultado de teste post-hoc encontrado (Friedman não foi significativo ou ocorreu um erro).",

        # PDF Strings
        "pdf_report_title": "RELATÓRIO DE AVALIAÇÃO DE MODELOS ML - {}",
        "pdf_report_subtitle": "Avaliação Nutricional Antropométrica",
        "pdf_equipment_heading": "CARACTERÍSTICAS DO EQUIPAMENTO DE PROCESSAMENTO",
        "pdf_component_header": "Componente",
        "pdf_specification_header": "Especificação",
        "pdf_processor": "Processador",
        "pdf_ram": "RAM instalada",
        "pdf_storage": "Armazenamento",
        "pdf_gpu": "Placa gráfica",
        "pdf_dataset_info_heading": "INFORMAÇÕES DO CONJUNTO DE DADOS",
        "pdf_num_records": "Número de registros: {}",
        "pdf_num_features": "Número de características: {}",
        "pdf_train_percent": "Porcentagem de treinamento: {:.2f}%",
        "pdf_test_percent": "Porcentagem de teste: {:.2f}%",
        "pdf_training_times_heading": "TEMPOS DE TREINAMENTO",
        "pdf_metrics_heading": "MÉTRICAS DE DESEMPENHO",
        "pdf_model_header": "Modelo",
        "pdf_time_seconds_header": "Tempo (segundos)",
        "pdf_mse": "MSE",
        "pdf_rmse": "RMSE",
        "pdf_mae": "MAE",
        "pdf_r2": "R²",
        "pdf_statistical_tests_heading": "TESTES ESTATÍSTICOS INFERENCIAIS",
        "pdf_residuals_normality_heading": "Teste de Normalidade dos Resíduos:",
        "pdf_shapiro_wilk_result": "{} (Shapiro-Wilk): p-valor = {:.4f} ({})",
        "pdf_kolmogorov_smirnov_result": "{} (Kolmogorov-Smirnov): p-valor = {:.4f} ({})",
        "pdf_shapiro_wilk_note": "{}: Shapiro-Wilk não pôde ser executado: {}",
        "pdf_zero_std_note": "{}: Resíduos com desvio padrão zero (todos os valores são iguais), o teste de normalidade não pode ser realizado.",
        "pdf_no_stats_found": "Nenhum resultado de teste estatístico encontrado para {}.",
        "pdf_additional_visualizations": "VISUALIZAÇÕES ADICIONAIS",
        "pdf_confusion_matrices": "Matrizes de Confusão",
        "pdf_confusion_matrix_for": "Matriz de Confusão para {}:",
        "pdf_confusion_matrix_warning": "Aviso: Matriz de Confusão para {} não encontrada em {}",
        "pdf_performance_graphs": "Gráficos de Desempenho do Modelo",
        "pdf_graphs_for_model": "Gráficos para {} ({}):",
        "pdf_graph_title_prefix": "- {}:",
        "pdf_graph_warning": "Aviso: Gráfico '{}' para {} não encontrado em {}",
        "pdf_target_suffix_warning": "Aviso: Nenhum sufixo encontrado para o alvo '{}'. Gráficos específicos não serão adicionados.",
        "pdf_residuals_graphs_heading": "GRÁFICOS DE RESÍDUOS"
    },
    "ko": { # Korean
        "page_title": "ML 모델 평가 - 영양 평가",
        "app_title": "🧬 인체 측정 영양 평가를 위한 ML 모델 평가",
        "app_description": "---",
        "sidebar_title": "⚙️ 설정",
        "select_language_label": "언어 선택:",
        "data_load_section": "데이터 로드",
        "data_loaded_success": "✅ 데이터 로드 성공",
        "records_label": "📊 기록:",
        "columns_label": "📋 열:",
        "file_not_found_error": "현재 폴더에서 'datos_pacientes2.csv' 파일을 찾을 수 없습니다.",
        "file_load_error": "파일 로드 오류: {}",
        "dataset_info_section": "📋 데이터셋 정보",
        "first_rows_label": "**첫 5행:**",
        "statistical_info_label": "**통계 정보:**",
        "target_variables_section": "🎯 대상 변수 평가",
        "start_evaluation_button": "🚀 모델 평가 시작",
        "preprocessing_spinner": "데이터 전처리 중...",
        "column_not_found_error": "처리된 데이터에서 열 '{}'을(를) 찾을 수 없습니다.",
        "data_split_section": "🗂️ {} 에 대한 데이터 분할",
        "train_percent": "**훈련 데이터 비율:** {}%",
        "test_percent": "**테스트 데이터 비율:** {}%",
        "model_training_section": "🤖 {} 에 대한 모델 훈련",
        "training_spinner": "{} 에 대한 모델 훈련 중...",
        "training_complete_success": "✅ {} 에 대한 모델 훈련 성공",
        "metrics_section": "📊 성능 지표",
        "training_times_section": "⏱️ 훈련 시간",
        "model_label": "모델",
        "time_seconds_label": "시간 (초)",
        "statistical_tests_section": "📈 추론 통계 테스트",
        "statistical_tests_spinner": "{} 에 대한 통계 테스트 수행 중...",
        "residuals_normality_test": "잔차 정규성 테스트:",
        "shapiro_wilk_test": "{} (Shapiro-Wilk): p-값 = {:.4f} ({})",
        "kolmogorov_smirnov_test": "{} (Kolmogorov-Smirnov): p-값 = {:.4f} ({})",
        "normal_interpretation": "✅ 정상",
        "not_normal_interpretation": "❌ 비정상",
        "shapiro_wilk_error": "{}: Shapiro-Wilk 을(를) 수행할 수 없습니다: {}",
        "zero_std_error": "{}: 잔차의 표준 편차가 0입니다 (모든 값이 동일함). 정규성 테스트를 수행할 수 없습니다.",
        "no_statistical_results": "{} 에 대한 통계 테스트 결과를 찾을 수 없습니다.",
        "evaluation_complete_success": "🎉 모든 대상 변수에 대한 모델 평가가 완료되었습니다.",
        "download_reports_section": "📄 PDF 보고서 다운로드",
        "download_pdf_link": "📥 {} 에 대한 PDF 보고서 다운로드",
        "pdf_generation_error": "{} 에 대한 PDF 보고서 생성 또는 다운로드 오류: {}",
        "dataset_load_error": "데이터셋을 로드할 수 없습니다. 'datos_pacientes.csv' 파일이 올바른 폴더에 있는지 확인하십시오.",
        "training_linear_regression": "🔄 선형 회귀 훈련 중...",
        "training_random_forest": "🔄 랜덤 포레스트 훈련 중...",
        "training_xgboost": "🔄 XGBoost 훈련 중...",
        "processed_columns": "처리된 열:",
        "residuals_histogram_title": "{} 에 대한 잔차 히스토그램",
        "qq_plot_title": "{} 에 대한 잔차 Q-Q 플롯",
        "residuals_vs_predictions_title": "{} 에 대한 잔차 대 예측",
        "residuals_label": "잔차",
        "frequency_label": "빈도",
        "predicted_values_label": "예측 값",
        "friedman_test_heading": "프리드만 테스트",
        "friedman_result": "프리드만 테스트 결과: 카이제곱 = {:.4f}, p-값 = {:.4f} ({})",
        "friedman_significant": "유의미",
        "friedman_not_significant_interpret": "유의미하지 않음",
        "friedman_not_enough_models": "프리드만 테스트를 실행하려면 최소 3개의 모델이 필요합니다.",
        "friedman_data_error": "프리드만 데이터 준비 중 오류: {}",
        "friedman_error": "프리드만 테스트 실행 중 오류: {}",
        "friedman_not_significant": "프리드만 테스트가 유의미하지 않아 사후 테스트가 수행되지 않았습니다.",
        "posthoc_heading": "사후 테스트 (Nemenyi)",
        "nemenyi_intro": "Nemenyi 사후 테스트 결과 (p-값):",
        "pdf_friedman_test_heading": "프리드만 테스트",
        "pdf_friedman_result": "카이제곱 통계량 = {:.4f}, p-값 = {:.4f} ({})",
        "pdf_posthoc_heading": "사후 테스트 (NEMENYI)",
        "pdf_nemenyi_intro": "Nemenyi 사후 테스트 결과 (p-값):",
        "pdf_no_friedman_results": "프리드만 테스트를 실행할 수 없었습니다.",
        "pdf_no_posthoc_results": "사후 테스트 결과를 찾을 수 없습니다 (프리드만 테스트가 유의미하지 않거나 오류가 발생했습니다).",

        # PDF Strings
        "pdf_report_title": "ML 모델 평가 보고서 - {}",
        "pdf_report_subtitle": "인체 측정 영양 평가",
        "pdf_equipment_heading": "처리 장비 특성",
        "pdf_component_header": "구성 요소",
        "pdf_specification_header": "사양",
        "pdf_processor": "프로세서",
        "pdf_ram": "설치된 RAM",
        "pdf_storage": "저장 공간",
        "pdf_gpu": "그래픽 카드",
        "pdf_dataset_info_heading": "데이터셋 정보",
        "pdf_num_records": "기록 수: {}",
        "pdf_num_features": "특징 수: {}",
        "pdf_train_percent": "훈련 비율: {:.2f}%",
        "pdf_test_percent": "테스트 비율: {:.2f}%",
        "pdf_training_times_heading": "훈련 시간",
        "pdf_metrics_heading": "성능 지표",
        "pdf_model_header": "모델",
        "pdf_time_seconds_header": "시간 (초)",
        "pdf_mse": "평균 제곱 오차 (MSE)",
        "pdf_rmse": "평균 제곱근 오차 (RMSE)",
        "pdf_mae": "평균 절대 오차 (MAE)",
        "pdf_r2": "결정 계수 (R²)",
        "pdf_statistical_tests_heading": "추론 통계 테스트",
        "pdf_residuals_normality_heading": "잔차 정규성 테스트:",
        "pdf_shapiro_wilk_result": "{} (Shapiro-Wilk): p-값 = {:.4f} ({})",
        "pdf_kolmogorov_smirnov_result": "{} (Kolmogorov-Smirnov): p-값 = {:.4f} ({})",
        "pdf_shapiro_wilk_note": "{}: Shapiro-Wilk 을(를) 수행할 수 없습니다: {}",
        "pdf_zero_std_note": "{}: 잔차의 표준 편차가 0입니다 (모든 값이 동일함). 정규성 테스트를 수행할 수 없습니다.",
        "pdf_no_stats_found": "{} 에 대한 통계 테스트 결과를 찾을 수 없습니다.",
        "pdf_additional_visualizations": "추가 시각화",
        "pdf_confusion_matrices": "혼동 행렬",
        "pdf_confusion_matrix_for": "{} 에 대한 혼동 행렬:",
        "pdf_confusion_matrix_warning": "경고: {} 에서 {} 에 대한 혼동 행렬을 찾을 수 없습니다",
        "pdf_performance_graphs": "모델 성능 그래프",
        "pdf_graphs_for_model": "{} ({}) 에 대한 그래프:",
        "pdf_graph_title_prefix": "- {}:",
        "pdf_graph_warning": "경고: {} 에서 {} 에 대한 '{}' 그래프를 찾을 수 없습니다",
        "pdf_target_suffix_warning": "경고: 대상 '{}' 에 대한 접미사를 찾을 수 없습니다. 특정 그래프는 추가되지 않습니다.",
        "pdf_residuals_graphs_heading": "잔차 그래프"
    }
}

# --- Initialize session state for language if not already set ---
if 'lang' not in st.session_state:
    st.session_state.lang = "es" # Default to Spanish

current_lang = LANGUAGES[st.session_state.lang]

# Configuración de la página (needs to be here to use current_lang)
st.set_page_config(
    page_title=current_lang["page_title"],
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title(current_lang["app_title"])
st.markdown(current_lang["app_description"])

# --- Funciones auxiliares ---
@st.cache_data
def load_data(current_lang):
    """Cargar datos del CSV"""
    try:
        df = pd.read_csv('datos_pacientes2.csv')
        return df
    except FileNotFoundError:
        st.error(current_lang["file_not_found_error"])
        return None
    except Exception as e:
        st.error(current_lang["file_load_error"].format(str(e)))
        return None

def preprocess_data(df, current_lang):
    """Preprocesamiento de datos"""
    df_processed = df.copy()

    # Codificar variables categóricas
    label_encoders = {}
    for column in df_processed.columns:
        if df_processed[column].dtype == 'object':
            le = LabelEncoder()
            df_processed[column] = le.fit_transform(df_processed[column].astype(str))
            label_encoders[column] = le

    # Manejar valores nulos
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
    
    st.write(current_lang["processed_columns"], df_processed.columns.tolist())
    
    return df_processed, label_encoders

def train_models(X_train, X_test, y_train, y_test, current_lang):
    """Entrenar los modelos y medir tiempos"""
    models = {}
    training_times = {}
    predictions = {}
    metrics = {}
    
    # Regresión Lineal
    st.write(current_lang["training_linear_regression"])
    start_time = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    training_times['Regresión Lineal'] = time.time() - start_time
    
    lr_pred = lr_model.predict(X_test)
    models['Regresión Lineal'] = lr_model
    predictions['Regresión Lineal'] = lr_pred
    
    # Random Forest
    st.write(current_lang["training_random_forest"])
    start_time = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    training_times['Random Forest'] = time.time() - start_time
    
    rf_pred = rf_model.predict(X_test)
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    
    # XGBoost
    st.write(current_lang["training_xgboost"])
    start_time = time.time()
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    training_times['XGBoost'] = time.time() - start_time
    
    xgb_pred = xgb_model.predict(X_test)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_pred
    
    # Calcular métricas
    for name, pred in predictions.items():
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
    
    return models, predictions, metrics, training_times

def statistical_tests(predictions, y_test, current_lang):
    """Realizar pruebas estadísticas inferenciales, incluyendo Friedman y post-hoc."""
    results = {}
    
    st.write(current_lang["statistical_tests_spinner"].format("")) # General message for tests
    
    # --- Pruebas de normalidad de residuos (ya existentes) ---
    for name, pred in predictions.items():
        residuals = y_test - pred
        
        if len(residuals) <= 5000 and len(residuals) > 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                results[name] = {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'test': 'Shapiro-Wilk'
                }
            except Exception as e:
                results[name] = {
                    'note': current_lang["shapiro_wilk_error"].format(name, str(e)),
                    'test': 'N/A'
                }
        else: # Para N > 5000, usamos Kolmogorov-Smirnov
            if residuals.std() > 0:
                ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
                results[name] = {
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'test': 'Kolmogorov-Smirnov'
                }
            else:
                results[name] = {
                    'note': current_lang["zero_std_error"].format(name),
                    'test': 'N/A'
                }
    
    # --- Prueba de Friedman y Post-Hoc ---
    model_names = list(predictions.keys())
    
    if len(model_names) >= 3: # Friedman requiere al menos 3 grupos
        try:
            # Step 1: Collect absolute errors for each model, ensuring they align by index
            # This is the correct way to build the DataFrame for scikit-posthocs
            df_errors_for_friedman = pd.DataFrame({
                name: np.abs(y_test - predictions[name]) for name in model_names
            })
            
            # Check if all columns have the same number of rows as y_test
            if not df_errors_for_friedman.empty and df_errors_for_friedman.shape[0] == len(y_test):
                # The data for scipy.stats.friedmanchisquare should be unpacked as separate arrays
                # You can get this by converting the DataFrame to a list of its column arrays
                data_for_scipy_friedman = [df_errors_for_friedman[col].values for col in df_errors_for_friedman.columns]

                friedman_stat, friedman_p = stats.friedmanchisquare(*data_for_scipy_friedman) # Unpack the list of arrays
                
                results['Friedman'] = {
                    'stat': friedman_stat,
                    'p_value': friedman_p,
                    'model_names': model_names # Guardar nombres para referencia
                }

                # If Friedman is significant, execute post-hoc tests (Nemenyi)
                if friedman_p < 0.05:
                    # sp.posthoc_nemenyi_friedman works directly with the DataFrame of errors
                    # Rows are observations, columns are groups (models)
                    posthoc_df = sp.posthoc_nemenyi_friedman(df_errors_for_friedman)
                    
                    posthoc_df.columns = model_names
                    posthoc_df.index = model_names
                    results['Nemenyi_posthoc'] = posthoc_df.to_string() # Guardar como string para el PDF
                else:
                    results['Nemenyi_posthoc'] = current_lang["friedman_not_significant"]

            else:
                results['Friedman'] = {
                    'note': current_lang["friedman_data_error"].format("longitudes de datos diferentes o DataFrame vacío")
                }

        except Exception as e:
            results['Friedman'] = {'note': current_lang["friedman_error"].format(str(e))}
    else:
        results['Friedman'] = {'note': current_lang["friedman_not_enough_models"]}

    return results


def add_images_to_pdf(story, styles, target_name, current_lang):
    """Agrega imágenes al PDF desde las rutas especificadas."""
    story.append(Spacer(1, 24))
    story.append(Paragraph(current_lang["pdf_additional_visualizations"], styles['Heading2']))
    
    model_folder_map = {
        'Regresión Lineal': 'RegresionLinealMulti',
        'Random Forest': 'RandomForest',
        'XGBoost': 'XGBoost'
    }

    target_suffix_map = {
        'Valoracion_Talla_Edad': 'talla_edad',
        'Valoracion_IMC_Talla': 'imc_talla'
    }
    
    current_target_suffix = target_suffix_map.get(target_name, '').lower()

    # --- Matrices de Confusión ---
    story.append(Spacer(1, 12))
    story.append(Paragraph(current_lang["pdf_confusion_matrices"], styles['Heading3']))
    
    confusion_matrix_files = {
        'Regresión Lineal': 'confusion_regresion.png',
        'Random Forest': 'confusion_randomforest.png',
        'XGBoost': 'confusion_xgboost.png'
    }

    for model_name, filename in confusion_matrix_files.items():
        filepath = os.path.join('confusion_matrices', filename)
        if os.path.exists(filepath):
            story.append(Paragraph(current_lang["pdf_confusion_matrix_for"].format(model_name), styles['Normal']))
            img = Image(filepath, width=3.5 * inch, height=3.0 * inch)
            story.append(img)
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(current_lang["pdf_confusion_matrix_warning"].format(model_name, filepath), styles['Normal']))
            story.append(Spacer(1, 6))

    # --- Gráficos de Rendimiento Específicos por Modelo ---
    story.append(Spacer(1, 18))
    story.append(Paragraph(current_lang["pdf_performance_graphs"], styles['Heading3']))

    for model_name, folder_name in model_folder_map.items():
        if current_target_suffix:
            story.append(Spacer(1, 12))
            story.append(Paragraph(current_lang["pdf_graphs_for_model"].format(model_name, target_name.replace('_', ' ')), styles['Heading4']))
            
            plot_types = [
                f'Curvas_precision_recall_multiclase_modelo_{current_target_suffix}.png',
                f'Curvas_roc_multiclase_modelo_{current_target_suffix}.png',
                f'Graficos_calibracion_{current_target_suffix}.png'
            ]

            for plot_file in plot_types:
                filepath = os.path.join('Graficas', folder_name, plot_file)
                if os.path.exists(filepath):
                    friendly_title = plot_file.replace('_', ' ').replace('.png', '').replace(f'modelo {current_target_suffix}', '').strip()
                    story.append(Paragraph(current_lang["pdf_graph_title_prefix"].format(friendly_title), styles['Normal']))
                    img = Image(filepath, width=5.0 * inch, height=3.5 * inch)
                    story.append(img)
                    story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(current_lang["pdf_graph_warning"].format(plot_file, model_name, filepath), styles['Normal']))
                    story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(current_lang["pdf_target_suffix_warning"].format(target_name), styles['Normal']))
            story.append(Spacer(1, 6))
    
    # --- NUEVO: Gráficos de Residuos ---
    story.append(Spacer(1, 18))
    story.append(Paragraph(current_lang["pdf_residuals_graphs_heading"], styles['Heading3']))

    # Mapeo de nombres de modelo a nombres de archivo para los nuevos gráficos de residuos
    # Asegúrate de que estos nombres de archivo coincidan con los generados en generate_residual_plots
    residual_plot_types = {
        'histograma_residuos': current_lang["residuals_histogram_title"].format(""),
        'qq_plot_residuos': current_lang["qq_plot_title"].format(""),
        'residuos_vs_predicciones': current_lang["residuals_vs_predictions_title"].format("")
    }

    if current_target_suffix: # Asegura que tenemos un sufijo de objetivo válido
        for model_name_raw, folder_name in model_folder_map.items():
            model_name_safe = model_name_raw.replace(" ", "_").lower()
            story.append(Spacer(1, 12))
            story.append(Paragraph(current_lang["pdf_graphs_for_model"].format(model_name_raw, target_name.replace('_', ' ')), styles['Heading4']))

            for plot_file_prefix, plot_title_template in residual_plot_types.items():
                filepath = os.path.join('Graficas', f'Residuals_{current_target_suffix}', folder_name, 
                                        f'{plot_file_prefix}_{model_name_safe}_{current_target_suffix}.png')
                
                if os.path.exists(filepath):
                    # Solo necesitas el nombre del modelo aquí para el título
                    friendly_title = plot_title_template.replace("{}", model_name_raw) 
                    story.append(Paragraph(current_lang["pdf_graph_title_prefix"].format(friendly_title), styles['Normal']))
                    img = Image(filepath, width=5.0 * inch, height=3.5 * inch)
                    story.append(img)
                    story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(current_lang["pdf_graph_warning"].format(f"{plot_file_prefix}_{model_name_safe}_{current_target_suffix}.png", model_name_raw, filepath), styles['Normal']))
                    story.append(Spacer(1, 6))
    else:
        story.append(Paragraph(current_lang["pdf_target_suffix_warning"].format(target_name), styles['Normal']))
        story.append(Spacer(1, 6))
    # --- FIN NUEVO ---

def generate_residual_plots(predictions, y_test, target_name, current_lang):
    """
    Genera histogramas, Q-Q plots y gráficos de residuos vs. predicciones
    para cada modelo y guarda las imágenes en la carpeta Graficas.
    """
    plots_saved_paths = []
    
    # Mapeo de nombres de modelo a nombres de carpeta
    model_folder_map = {
        'Regresión Lineal': 'RegresionLinealMulti',
        'Random Forest': 'RandomForest',
        'XGBoost': 'XGBoost'
    }

    # Definir sufijo para las carpetas dentro de Graficas
    target_suffix = {
        'Valoracion_Talla_Edad': 'talla_edad',
        'Valoracion_IMC_Talla': 'imc_talla'
    }.get(target_name, target_name.lower().replace('_', ''))

    base_dir = os.path.join('Graficas', f'Residuals_{target_suffix}')
    os.makedirs(base_dir, exist_ok=True) # Asegurarse de que la carpeta base exista

    for model_name, pred in predictions.items():
        residuals = y_test - pred
        
        # Obtener la carpeta específica del modelo dentro de 'Graficas'
        model_sub_folder = model_folder_map.get(model_name, model_name.replace(' ', ''))
        save_dir = os.path.join(base_dir, model_sub_folder)
        os.makedirs(save_dir, exist_ok=True) # Crear subcarpeta para cada modelo

        # 1. Histograma de Residuos
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.title(f'{current_lang["pdf_report_title"].format(target_name.replace("_", " "))}\n{current_lang["residuals_histogram_title"].format(model_name)}')
        plt.xlabel(current_lang["residuals_label"])
        plt.ylabel(current_lang["frequency_label"])
        hist_path = os.path.join(save_dir, f'histograma_residuos_{model_name.replace(" ", "_").lower()}_{target_suffix}.png')
        plt.savefig(hist_path)
        plt.close()
        plots_saved_paths.append(hist_path)

        # 2. Gráfico Q-Q
        plt.figure(figsize=(8, 6))
        sm.qqplot(residuals, line='s', fit=True)
        plt.title(f'{current_lang["pdf_report_title"].format(target_name.replace("_", " "))}\n{current_lang["qq_plot_title"].format(model_name)}')
        qq_path = os.path.join(save_dir, f'qq_plot_residuos_{model_name.replace(" ", "_").lower()}_{target_suffix}.png')
        plt.savefig(qq_path)
        plt.close()
        plots_saved_paths.append(qq_path)

        # 3. Gráfico de Residuos vs. Predicciones
        plt.figure(figsize=(8, 6))
        plt.scatter(pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.title(f'{current_lang["pdf_report_title"].format(target_name.replace("_", " "))}\n{current_lang["residuals_vs_predictions_title"].format(model_name)}')
        plt.xlabel(current_lang["predicted_values_label"])
        plt.ylabel(current_lang["residuals_label"])
        plt.grid(True)
        residuals_vs_pred_path = os.path.join(save_dir, f'residuos_vs_predicciones_{model_name.replace(" ", "_").lower()}_{target_suffix}.png')
        plt.savefig(residuals_vs_pred_path)
        plt.close()
        plots_saved_paths.append(residuals_vs_pred_path)
        
    return plots_saved_paths

def create_pdf_report(metrics, training_times, statistical_results, df_info, target_name, train_size_percent, test_size_percent, current_lang):
    """Crear reporte PDF, incluyendo Friedman y post-hoc."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Título (sin cambios)
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1
    )
    
    story.append(Paragraph(current_lang["pdf_report_title"].format(target_name.replace('_', ' ').upper()), title_style))
    story.append(Paragraph(current_lang["pdf_report_subtitle"], styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Información del equipo (sin cambios)
    story.append(Paragraph(current_lang["pdf_equipment_heading"], styles['Heading2']))
    equipment_data = [
        [current_lang["pdf_component_header"], current_lang["pdf_specification_header"]],
        [current_lang["pdf_processor"], 'Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz'],
        [current_lang["pdf_ram"], '8,00 GB (7,83 GB usable)'],
        [current_lang["pdf_storage"], '932 GB HDD, 238 GB SSD NVMe'],
        [current_lang["pdf_gpu"], 'NVIDIA GeForce GTX 1650 (4 GB)']
    ]
    
    equipment_table = Table(equipment_data, colWidths=[2*inch, 4*inch])
    equipment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(equipment_table)
    story.append(Spacer(1, 12))
    
    # Información del dataset (sin cambios)
    story.append(Paragraph(current_lang["pdf_dataset_info_heading"], styles['Heading2']))
    story.append(Paragraph(current_lang["pdf_num_records"].format(df_info['rows']), styles['Normal']))
    story.append(Paragraph(current_lang["pdf_num_features"].format(df_info['columns']), styles['Normal']))
    story.append(Paragraph(current_lang["pdf_train_percent"].format(train_size_percent), styles['Normal']))
    story.append(Paragraph(current_lang["pdf_test_percent"].format(test_size_percent), styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Tiempos de entrenamiento (sin cambios)
    story.append(Paragraph(current_lang["pdf_training_times_heading"], styles['Heading2']))
    time_data = [[current_lang["pdf_model_header"], current_lang["pdf_time_seconds_header"]]]
    for model, time_val in training_times.items():
        time_data.append([model, f"{time_val:.4f}"])
    
    time_table = Table(time_data, colWidths=[3*inch, 2*inch])
    time_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(time_table)
    story.append(Spacer(1, 12))
    
    # Métricas de rendimiento (sin cambios)
    story.append(Paragraph(current_lang["pdf_metrics_heading"], styles['Heading2']))
    metrics_data = [[current_lang["pdf_model_header"], current_lang["pdf_mse"], current_lang["pdf_rmse"], current_lang["pdf_mae"], current_lang["pdf_r2"]]]
    for model, metric in metrics.items():
        metrics_data.append([
            model,
            f"{metric['MSE']:.4f}",
            f"{metric['RMSE']:.4f}",
            f"{metric['MAE']:.4f}",
            f"{metric['R²']:.4f}"
        ])
    
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 12))
    
    # Pruebas estadísticas (modificado para Friedman)
    story.append(Paragraph(current_lang["pdf_statistical_tests_heading"], styles['Heading2']))
    
    story.append(Paragraph(current_lang["pdf_residuals_normality_heading"], styles['Heading3']))
    for model_name_for_stats in metrics.keys(): # Iterar sobre los modelos para la sección de normalidad
        if model_name_for_stats in statistical_results:
            if 'note' in statistical_results[model_name_for_stats]:
                story.append(Paragraph(statistical_results[model_name_for_stats]['note'], styles['Normal']))
            elif statistical_results[model_name_for_stats]['test'] == 'Shapiro-Wilk':
                p_val = statistical_results[model_name_for_stats]['shapiro_p']
                interpretation = current_lang["normal_interpretation"] if p_val > 0.05 else current_lang["not_normal_interpretation"]
                story.append(Paragraph(current_lang["pdf_shapiro_wilk_result"].format(model_name_for_stats, p_val, interpretation), styles['Normal']))
            else: # Kolmogorov-Smirnov
                p_val = statistical_results[model_name_for_stats]['ks_p']
                interpretation = current_lang["normal_interpretation"] if p_val > 0.05 else current_lang["not_normal_interpretation"]
                story.append(Paragraph(current_lang["pdf_kolmogorov_smirnov_result"].format(model_name_for_stats, p_val, interpretation), styles['Normal']))
        else:
            story.append(Paragraph(current_lang["pdf_no_stats_found"].format(model_name_for_stats), styles['Normal']))
    story.append(Spacer(1, 12))

    # --- NUEVA SECCIÓN: Prueba de Friedman y Post-Hoc ---
    story.append(Paragraph(current_lang["pdf_friedman_test_heading"], styles['Heading3']))
    if 'Friedman' in statistical_results:
        friedman_res = statistical_results['Friedman']
        if 'note' in friedman_res:
            story.append(Paragraph(friedman_res['note'], styles['Normal']))
        else:
            story.append(Paragraph(current_lang["pdf_friedman_result"].format(
                friedman_res['stat'], friedman_res['p_value'], # Pass the values directly
                current_lang["friedman_significant"] if friedman_res['p_value'] < 0.05 else current_lang["friedman_not_significant_interpret"]
            ), styles['Normal']))
            
            story.append(Spacer(1, 6))
            story.append(Paragraph(current_lang["pdf_posthoc_heading"], styles['Heading4']))
            if 'Nemenyi_posthoc' in statistical_results:
                story.append(Paragraph(current_lang["pdf_nemenyi_intro"], styles['Normal']))
                # Use Preformatted to keep the table-like structure of the string
                story.append(Preformatted(statistical_results['Nemenyi_posthoc'], styles['Code']))
            else:
                story.append(Paragraph(current_lang["pdf_no_posthoc_results"], styles['Normal']))
    else:
        story.append(Paragraph(current_lang["pdf_no_friedman_results"], styles['Normal']))
    story.append(Spacer(1, 12))
    # --- FIN NUEVA SECCIÓN ---

    # --- AGREGAR IMÁGENES AL PDF (sin cambios) ---
    add_images_to_pdf(story, styles, target_name, current_lang) # Pass current_lang to image function

    doc.build(story)
    buffer.seek(0)
    return buffer

# Interfaz principal
def main():
    # Update current_lang at the beginning of main to reflect any language changes from sidebar
    global current_lang
    # Ensure 'lang' is initialized in session_state, defaulting to 'es' if not set
    if 'lang' not in st.session_state:
        st.session_state.lang = "es"
        
    current_lang = LANGUAGES[st.session_state.lang]

    st.sidebar.title(current_lang["sidebar_title"])
    
    # Define the language options to be displayed and their corresponding internal keys
    language_options_display = {
        "Español": "es",
        "English": "en",
        "中文 (Chino)": "zh",
        "Deutsch (Alemán)": "de",
        "日本語 (Japonés)": "ja",
        "Français (Francés)": "fr",
        "Português (Portugués)": "pt",
        "한국어 (Coreano)": "ko"
    }

    # Get the display name for the current language key in session state
    current_lang_display = next(
        (display_name for display_name, lang_key in language_options_display.items()
         if lang_key == st.session_state.lang),
        "Español" # Default to 'Español' if the current lang key isn't found in display options
    )

    # Language selection selectbox
    language_selection_display = st.sidebar.selectbox(
        current_lang["select_language_label"], # This label should be defined in your LANGUAGES dict
        options=list(language_options_display.keys()),
        index=list(language_options_display.keys()).index(current_lang_display),
        key="language_selection"
    )

    # Get the internal language key based on the user's selection
    selected_lang_key = language_options_display[language_selection_display]

    # If the selected language is different from the current one, update and rerun
    if selected_lang_key != st.session_state.lang:
        st.session_state.lang = selected_lang_key
        st.rerun() # Rerun to update all texts
            
    # Re-fetch current_lang after potential rerun (ensures current_lang is correct after a language change)
    current_lang = LANGUAGES[st.session_state.lang]

    # Cargar datos
    st.subheader(current_lang["data_load_section"])
    df = load_data(current_lang)
    
    if df is not None:
        st.sidebar.success(current_lang["data_loaded_success"])
        st.sidebar.write(current_lang["records_label"].format(df.shape[0]))
        st.sidebar.write(current_lang["columns_label"].format(df.shape[1]))
        
        # Mostrar información del dataset
        st.subheader(current_lang["dataset_info_section"])
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(current_lang["first_rows_label"])
            st.dataframe(df.head())
        
        with col2:
            st.write(current_lang["statistical_info_label"])
            st.dataframe(df.describe())
        
        # Selección de variables objetivo
        st.subheader(current_lang["target_variables_section"])
        target_columns = ['Valoracion_Talla_Edad', 'Valoracion_IMC_Talla']

        if "evaluated_data" not in st.session_state:
            st.session_state.evaluated_data = {}

        if st.button(current_lang["start_evaluation_button"], type="primary"):
            st.session_state.evaluated_data = {} # Reset previous data
            # Preprocesar datos
            with st.spinner(current_lang["preprocessing_spinner"]):
                df_processed, label_encoders = preprocess_data(df, current_lang)
                
                # Verificar si las columnas objetivo están en el DataFrame procesado
                for target_column in target_columns:
                    if target_column not in df_processed.columns:
                        st.error(current_lang["column_not_found_error"].format(target_column))
                        return # Exit if column not found

                    # Separar características y variable objetivo
                    X = df_processed.drop(columns=[target_column])
                    y = df_processed[target_column]
                    
                    # División train/test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Calculate and display train/test percentages
                    train_size_percent = (len(X_train) / len(df_processed)) * 100
                    test_size_percent = (len(X_test) / len(df_processed)) * 100
                    
                    st.write(f"---")
                    st.subheader(current_lang["data_split_section"].format(target_column))
                    st.write(current_lang["train_percent"].format(train_size_percent))
                    st.write(current_lang["test_percent"].format(test_size_percent))
                    st.write(f"---")

                    # Entrenar modelos
                    st.subheader(current_lang["model_training_section"].format(target_column))
                    with st.spinner(current_lang["training_spinner"].format(target_column)):
                        models, predictions, metrics, training_times = train_models(
                            X_train, X_test, y_train, y_test, current_lang
                        )
                    
                    st.success(current_lang["training_complete_success"].format(target_column))
                    
                    # Store results in session state
                    st.session_state.evaluated_data[target_column] = {
                        'metrics': metrics,
                        'training_times': training_times,
                        'predictions': predictions,
                        'y_test': y_test, # Store y_test for statistical tests later
                        'train_size_percent': train_size_percent,
                        'test_size_percent': test_size_percent
                    }
                    
                    # Display metrics immediately after training for visual feedback
                    st.subheader(current_lang["metrics_section"])
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(metrics_df.style.highlight_min(axis=0, subset=['MSE', 'RMSE', 'MAE']).highlight_max(axis=0, subset=['R²']))
                    
                    # Tiempos de entrenamiento
                    st.subheader(current_lang["training_times_section"])
                    time_df = pd.DataFrame(list(training_times.items()), columns=[current_lang["model_label"], current_lang["time_seconds_label"]])
                    st.dataframe(time_df)
                    
                    # Pruebas estadísticas
                    st.subheader(current_lang["statistical_tests_section"])
                    with st.spinner(current_lang["statistical_tests_spinner"].format(target_column)):
                        statistical_results = statistical_tests(predictions, y_test, current_lang)
                    
                    # Display statistical results
                    st.session_state.evaluated_data[target_column]['statistical_results'] = statistical_results
                    for model in models.keys():
                        if model in statistical_results:
                            if 'note' in statistical_results[model]:
                                st.write(f"- {statistical_results[model]['note']}")
                            elif statistical_results[model]['test'] == 'Shapiro-Wilk':
                                p_val = statistical_results[model]['shapiro_p']
                                interpretation = current_lang["normal_interpretation"] if p_val > 0.05 else current_lang["not_normal_interpretation"]
                                st.write(current_lang["shapiro_wilk_test"].format(model, p_val, interpretation))
                            else:
                                p_val = statistical_results[model]['ks_p']
                                interpretation = current_lang["normal_interpretation"] if p_val > 0.05 else current_lang["not_normal_interpretation"]
                                st.write(current_lang["kolmogorov_smirnov_test"].format(model, p_val, interpretation))
                        else:
                            st.write(current_lang["no_statistical_results"].format(model))
                    # --- NUEVO: Generar y guardar gráficos de residuos ---
                    # Guardamos las rutas de los gráficos generados para cada objetivo
                    st.session_state.evaluated_data[target_column]['residual_plots_paths'] = generate_residual_plots(
                        predictions, y_test, target_column, current_lang
                    )
            st.success(current_lang["evaluation_complete_success"])
            st.markdown("---") # Separator after all training is done

        # Display download buttons after all evaluations are done and stored in session_state
        if st.session_state.evaluated_data:
            st.subheader(current_lang["download_reports_section"])
            df_info = {'rows': df.shape[0], 'columns': df.shape[1]}
            for target_column, data in st.session_state.evaluated_data.items():
                metrics = data['metrics']
                training_times = data['training_times']
                statistical_results = data['statistical_results'] # Retrieved from session state
                train_size_percent = data['train_size_percent']
                test_size_percent = data['test_size_percent']

                try:
                    pdf_buffer = create_pdf_report(metrics, training_times, statistical_results, df_info, target_column, train_size_percent, test_size_percent, current_lang)
                    pdf_bytes = pdf_buffer.getvalue()
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="reporte_evaluacion_modelos_{target_column}.pdf">{current_lang["download_pdf_link"].format(target_column.replace("_", " "))}</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(current_lang["pdf_generation_error"].format(target_column, str(e)))

    else:
        st.error(current_lang["dataset_load_error"])

if __name__ == "__main__":
    main()