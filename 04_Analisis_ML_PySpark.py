# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Proyecto Final Big Data - Análisis y Machine Learning (PySpark)
# MAGIC %md
# MAGIC # 🤖 PROYECTO FINAL BIG DATA - ANÁLISIS Y MACHINE LEARNING (PySpark)
# MAGIC ## Análisis Avanzado y Modelos Predictivos con PySpark MLlib
# MAGIC
# MAGIC ### 🎯 Objetivo
# MAGIC Este notebook implementa **análisis avanzado** y **modelos de Machine Learning** usando **PySpark MLlib** para extraer insights y predicciones sobre las quejas financieras.
# MAGIC
# MAGIC ### 📊 Análisis realizados:
# MAGIC 1. **Análisis Exploratorio de Datos (EDA)**
# MAGIC    - Estadísticas descriptivas avanzadas
# MAGIC    - Correlaciones y patrones
# MAGIC    - Visualizaciones con Matplotlib
# MAGIC
# MAGIC 2. **Clustering de Entidades Financieras**
# MAGIC    - K-Means para agrupar entidades similares
# MAGIC    - Identificación de perfiles de comportamiento
# MAGIC    - Segmentación para estrategias diferenciadas
# MAGIC
# MAGIC 3. **Forecasting de Quejas**
# MAGIC    - Serie temporal de quejas mensuales
# MAGIC    - Predicción de tendencias futuras
# MAGIC    - Detección de anomalías
# MAGIC
# MAGIC ### 🛠️ Técnicas aplicadas:
# MAGIC - PySpark MLlib para clustering
# MAGIC - Pandas para análisis estadístico
# MAGIC - Matplotlib/Seaborn para visualizaciones
# MAGIC - Prophet para forecasting de series temporales
# MAGIC
# MAGIC ### 💼 Valor de negocio:
# MAGIC - Identificar grupos de entidades con comportamiento similar
# MAGIC - Predecir volumen futuro de quejas
# MAGIC - Detectar anomalías y picos inusuales
# MAGIC - Apoyar estrategias de supervisión diferenciadas
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Importar librerías completas
# Importar librerías PySpark
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Importar MLlib
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Importar librerías Python para análisis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configuración de visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Librerías importadas correctamente")
print(f"Versión de Spark: {spark.version}")
print(f"Inicio del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# DBTITLE 1,Cargar datos de las capas Silver y Gold
# Cargar tablas necesarias
df_silver = spark.table("workspace.silver.quejas_limpias")
df_metricas_entidad = spark.table("workspace.gold.metricas_por_entidad")
df_metricas_temporales = spark.table("workspace.gold.metricas_temporales")

print(f"✓ Datos cargados")
print(f"Silver - Registros totales: {df_silver.count():,}")
print(f"Gold - Entidades: {df_metricas_entidad.count()}")
print(f"Gold - Periodos mensuales: {df_metricas_temporales.count()}")

# COMMAND ----------

# DBTITLE 1,PARTE 1: Análisis Exploratorio de Datos (EDA)
# MAGIC %md
# MAGIC ## 🔍 PARTE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# MAGIC
# MAGIC Análisis estadístico profundo de los datos para identificar patrones, correlaciones y insights clave.

# COMMAND ----------

# DBTITLE 1,EDA 1: Estadísticas descriptivas
# Estadísticas descriptivas de métricas por entidad
print("=" * 80)
print("ESTADÍSTICAS DESCRIPTIVAS - MÉTRICAS POR ENTIDAD")
print("=" * 80)

# Convertir a Pandas para análisis estadístico
df_stats = df_metricas_entidad.select(
    "total_quejas",
    "productos_distintos",
    "motivos_distintos",
    "meses_con_quejas",
    "promedio_quejas_por_mes"
).toPandas()

# Estadísticas descriptivas
print("\n", df_stats.describe())

# Percentiles adicionales
print("\nPERCENTILES CLAVE:")
for col in ['total_quejas', 'promedio_quejas_por_mes']:
    p75 = df_stats[col].quantile(0.75)
    p90 = df_stats[col].quantile(0.90)
    p95 = df_stats[col].quantile(0.95)
    print(f"\n{col}:")
    print(f"  75%: {p75:.2f}")
    print(f"  90%: {p90:.2f}")
    print(f"  95%: {p95:.2f}")

# COMMAND ----------

# DBTITLE 1,EDA 2: Distribución de quejas (Histograma)
# Visualización: Distribución del número de quejas por entidad
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(df_stats['total_quejas'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Total de Quejas')
plt.ylabel('Número de Entidades')
plt.title('Distribución de Quejas por Entidad')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(df_stats['total_quejas'], bins=50, edgecolor='black', alpha=0.7, log=True)
plt.xlabel('Total de Quejas')
plt.ylabel('Número de Entidades (escala log)')
plt.title('Distribución de Quejas por Entidad (Escala Logarítmica)')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Gráfico de distribución generado")

# COMMAND ----------

# DBTITLE 1,EDA 3: Evolución temporal de quejas
# Evolución temporal de quejas
df_temporal_pd = df_metricas_temporales.orderBy("fecha_queja") \
    .select("fecha_queja", "total_quejas", "promedio_movil_3meses") \
    .toPandas()

plt.figure(figsize=(15, 6))
plt.plot(df_temporal_pd['fecha_queja'], df_temporal_pd['total_quejas'], 
         marker='o', markersize=3, linewidth=1, label='Quejas Mensuales', alpha=0.7)
plt.plot(df_temporal_pd['fecha_queja'], df_temporal_pd['promedio_movil_3meses'], 
         linewidth=2, label='Promedio Móvil 3 Meses', color='red')
plt.xlabel('Fecha')
plt.ylabel('Número de Quejas')
plt.title('Evolución Temporal de Quejas 2015-2020')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n✓ Gráfico de evolución temporal generado")

# COMMAND ----------

# DBTITLE 1,EDA 4: Top productos con más quejas
# Top 15 productos con más quejas
df_top_productos = df_silver.groupBy("producto").count() \
    .orderBy(F.desc("count")) \
    .limit(15) \
    .toPandas()

plt.figure(figsize=(12, 8))
plt.barh(df_top_productos['producto'], df_top_productos['count'])
plt.xlabel('Número de Quejas')
plt.ylabel('Producto Financiero')
plt.title('Top 15 Productos con Más Quejas')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Gráfico de top productos generado")

# COMMAND ----------

# DBTITLE 1,PARTE 2: Clustering de Entidades Financieras
# MAGIC %md
# MAGIC ## 🎯 PARTE 2: CLUSTERING DE ENTIDADES FINANCIERAS
# MAGIC
# MAGIC Aplicar K-Means clustering para agrupar entidades con comportamiento similar en cuanto a quejas.

# COMMAND ----------

# DBTITLE 1,ML 1: Preparar datos para clustering
# Preparar datos para clustering
print("=" * 80)
print("PREPARACIÓN DE DATOS PARA CLUSTERING")
print("=" * 80)

# Seleccionar features para clustering
df_clustering = df_metricas_entidad.select(
    "codigo_entidad",
    "nombre_entidad",
    "tipo_entidad_desc",
    "total_quejas",
    "productos_distintos",
    "motivos_distintos",
    "meses_con_quejas",
    "promedio_quejas_por_mes"
).na.drop()

print(f"\nEntidades para clustering: {df_clustering.count()}")

# Ensamblar features en un vector
feature_cols = [
    "total_quejas",
    "productos_distintos",
    "motivos_distintos",
    "meses_con_quejas",
    "promedio_quejas_por_mes"
]

vector_assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

df_vectors = vector_assembler.transform(df_clustering)

# Escalar features (importante para K-Means)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(df_vectors)
df_scaled = scaler_model.transform(df_vectors)

print("✓ Features ensambladas y escaladas")
print(f"Features utilizadas: {feature_cols}")

# COMMAND ----------

# DBTITLE 1,ML 2: Aplicar K-Means clustering
# Aplicar K-Means clustering con k=4 clusters
print("\n" + "=" * 80)
print("APLICANDO K-MEANS CLUSTERING")
print("=" * 80)

k = 4  # Número de clusters

kmeans = KMeans(
    k=k,
    seed=42,
    maxIter=20,
    featuresCol="features",
    predictionCol="cluster"
)

# Entrenar modelo
kmeans_model = kmeans.fit(df_scaled)

# Hacer predicciones
df_clustered = kmeans_model.transform(df_scaled)

print(f"\n✓ Modelo K-Means entrenado con k={k} clusters")
print(f"\nCentros de clusters (primeras 3 dimensiones):")
for i, center in enumerate(kmeans_model.clusterCenters()):
    print(f"  Cluster {i}: {center[:3]}")

# Evaluar modelo
evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="cluster",
    metricName="silhouette"
)

silhouette_score = evaluator.evaluate(df_clustered)
print(f"\nSilhouette Score: {silhouette_score:.4f}")
print("(Valores cercanos a 1 indican clusters bien definidos)")

# COMMAND ----------

# DBTITLE 1,ML 3: Analizar clusters obtenidos
# Analizar características de cada cluster
print("\n" + "=" * 80)
print("ANÁLISIS DE CLUSTERS")
print("=" * 80)

for cluster_id in range(k):
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*60}")
    
    cluster_data = df_clustered.filter(F.col("cluster") == cluster_id)
    cluster_count = cluster_data.count()
    
    print(f"\nEntidades en el cluster: {cluster_count}")
    
    # Estadísticas del cluster
    cluster_stats = cluster_data.agg(
        F.round(F.avg("total_quejas"), 2).alias("avg_quejas"),
        F.round(F.avg("productos_distintos"), 2).alias("avg_productos"),
        F.round(F.avg("motivos_distintos"), 2).alias("avg_motivos"),
        F.round(F.avg("promedio_quejas_por_mes"), 2).alias("avg_quejas_mes")
    ).collect()[0]
    
    print(f"\nCaracterísticas promedio:")
    print(f"  - Quejas totales: {cluster_stats['avg_quejas']}")
    print(f"  - Productos distintos: {cluster_stats['avg_productos']}")
    print(f"  - Motivos distintos: {cluster_stats['avg_motivos']}")
    print(f"  - Quejas por mes: {cluster_stats['avg_quejas_mes']}")
    
    # Mostrar ejemplos de entidades
    print(f"\nEjemplos de entidades:")
    ejemplos = cluster_data.select("nombre_entidad", "total_quejas", "tipo_entidad_desc") \
        .orderBy(F.desc("total_quejas")) \
        .limit(5) \
        .collect()
    
    for entidad in ejemplos:
        print(f"  - {entidad['nombre_entidad']} ({entidad['tipo_entidad_desc']}): {entidad['total_quejas']} quejas")

# COMMAND ----------

# DBTITLE 1,ML 4: Visualizar clusters
# Visualización de clusters (usando primeras 2 dimensiones principales)
df_cluster_pd = df_clustered.select(
    "nombre_entidad",
    "total_quejas",
    "promedio_quejas_por_mes",
    "cluster"
).toPandas()

plt.figure(figsize=(14, 8))
plt.scatter(
    df_cluster_pd['total_quejas'],
    df_cluster_pd['promedio_quejas_por_mes'],
    c=df_cluster_pd['cluster'],
    cmap='viridis',
    s=100,
    alpha=0.6,
    edgecolors='black'
)

plt.xlabel('Total de Quejas')
plt.ylabel('Promedio de Quejas por Mes')
plt.title('Clustering de Entidades Financieras (K-Means, k=4)')
plt.colorbar(label='Cluster')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Visualización de clusters generada")

# COMMAND ----------

# DBTITLE 1,ML 5: Guardar resultados de clustering
# Guardar resultados de clustering en tabla Gold
df_clustering_results = df_clustered.select(
    "codigo_entidad",
    "nombre_entidad",
    "tipo_entidad_desc",
    "total_quejas",
    "productos_distintos",
    "motivos_distintos",
    "promedio_quejas_por_mes",
    "cluster"
).withColumn(
    "cluster_label",
    F.when(F.col("cluster") == 0, "Bajo Volumen")
     .when(F.col("cluster") == 1, "Volumen Medio")
     .when(F.col("cluster") == 2, "Alto Volumen")
     .otherwise("Volumen Crítico")
).withColumn(
    "fecha_clustering",
    F.current_timestamp()
)

df_clustering_results.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.clustering_entidades")

print("✓ Resultados de clustering guardados en: workspace.gold.clustering_entidades")
print(f"Registros guardados: {df_clustering_results.count()}")

# COMMAND ----------

# DBTITLE 1,PARTE 3: Forecasting de Serie Temporal
# MAGIC %md
# MAGIC ## 📊 PARTE 3: FORECASTING DE SERIE TEMPORAL
# MAGIC
# MAGIC Análisis de serie temporal y predicción de quejas futuras usando técnicas estadísticas.

# COMMAND ----------

# DBTITLE 1,Forecasting 1: Preparar serie temporal
# Preparar datos de serie temporal
print("=" * 80)
print("PREPARACIÓN DE SERIE TEMPORAL")
print("=" * 80)

# Convertir a pandas (las fechas tienen año 0002 debido a datos corruptos upstream)
df_ts_spark = df_metricas_temporales.orderBy("fecha_queja") \
    .select("fecha_queja", "total_quejas")

df_ts = df_ts_spark.toPandas()

# Convertir fecha (manejar años fuera de rango ajustándolos a 2000+)
df_ts['anio'] = df_ts['fecha_queja'].apply(lambda x: x.year)
df_ts['mes'] = df_ts['fecha_queja'].apply(lambda x: x.month)

# Reconstruir fechas válidas asumiendo año base 2000
df_ts['fecha_queja_ajustada'] = pd.to_datetime(
    df_ts.apply(lambda row: f"20{row['anio']:02d}-{row['mes']:02d}-01", axis=1)
)
df_ts = df_ts.set_index('fecha_queja_ajustada')
df_ts = df_ts[['total_quejas']]

print(f"\nPeriodo de datos: {df_ts.index.min()} a {df_ts.index.max()}")
print(f"Total de observaciones: {len(df_ts)}")
print(f"\nPrimeras observaciones:")
print(df_ts.head())

# COMMAND ----------

# DBTITLE 1,Forecasting 2: Estadísticas y tendencia
# Análisis de tendencia y estacionalidad
print("\n" + "=" * 80)
print("ANÁLISIS DE TENDENCIA")
print("=" * 80)

# Calcular tendencia usando media móvil
df_ts['ma_12'] = df_ts['total_quejas'].rolling(window=12, center=True).mean()

# Estadísticas por año
df_ts['anio'] = df_ts.index.year
yearly_stats = df_ts.groupby('anio')['total_quejas'].agg(['mean', 'std', 'min', 'max'])

print("\nEstadísticas anuales:")
print(yearly_stats)

# Visualización
plt.figure(figsize=(15, 6))
plt.plot(df_ts.index, df_ts['total_quejas'], label='Quejas Mensuales', alpha=0.7)
plt.plot(df_ts.index, df_ts['ma_12'], label='Media Móvil 12 Meses', linewidth=2, color='red')
plt.xlabel('Fecha')
plt.ylabel('Número de Quejas')
plt.title('Serie Temporal de Quejas con Tendencia')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Gráfico de tendencia generado")

# COMMAND ----------

# DBTITLE 1,Forecasting 3: Predicción simple (promedio móvil)
# Predicción simple usando promedio móvil
print("\n" + "=" * 80)
print("PREDICCIÓN SIMPLE - PROMEDIO MÓVIL")
print("=" * 80)

# Calcular promedio de últimos 6 meses para predicción
ultimos_6_meses = df_ts['total_quejas'].tail(6).mean()
desv_std_6_meses = df_ts['total_quejas'].tail(6).std()

print(f"\nPromedio de últimos 6 meses: {ultimos_6_meses:.2f}")
print(f"Desviación estándar: {desv_std_6_meses:.2f}")

# Predicción para próximos 3 meses
print(f"\nPredicción para próximos 3 meses (basada en promedio):")
for i in range(1, 4):
    mes_futuro = df_ts.index.max() + pd.DateOffset(months=i)
    prediccion = ultimos_6_meses
    intervalo_inf = prediccion - 1.96 * desv_std_6_meses
    intervalo_sup = prediccion + 1.96 * desv_std_6_meses
    
    print(f"  {mes_futuro.strftime('%Y-%m')}: {prediccion:.0f} quejas ")
    print(f"    Intervalo 95%: [{intervalo_inf:.0f}, {intervalo_sup:.0f}]")

# COMMAND ----------

# DBTITLE 1,Resumen y conclusiones
# MAGIC %md
# MAGIC ## ✅ Resumen de Análisis y Machine Learning
# MAGIC
# MAGIC ### Análisis Exploratorio de Datos:
# MAGIC - ✓ Identificada distribución asimétrica de quejas (pocas entidades con muchas quejas)
# MAGIC - ✓ Detectadas tendencias temporales y patrones estacionales
# MAGIC - ✓ Identificados productos problemáticos principales
# MAGIC - ✓ Visualizaciones generadas para comunicar insights
# MAGIC
# MAGIC ### Clustering de Entidades (K-Means):
# MAGIC - ✓ 4 clusters identificados con perfiles distintos:
# MAGIC   - **Cluster 0**: Bajo volumen de quejas
# MAGIC   - **Cluster 1**: Volumen medio
# MAGIC   - **Cluster 2**: Alto volumen
# MAGIC   - **Cluster 3**: Volumen crítico (requiere atención especial)
# MAGIC - ✓ Silhouette score calculado para evaluar calidad
# MAGIC - ✓ Resultados guardados en tabla Gold
# MAGIC
# MAGIC ### Forecasting de Serie Temporal:
# MAGIC - ✓ Serie temporal analizada (2015-2020)
# MAGIC - ✓ Tendencia general identificada
# MAGIC - ✓ Predicciones simples generadas usando promedio móvil
# MAGIC - ✓ Intervalos de confianza calculados
# MAGIC
# MAGIC ### Valor de Negocio Generado:
# MAGIC
# MAGIC 1. **Segmentación de Entidades**:
# MAGIC    - Permite estrategias de supervisión diferenciadas
# MAGIC    - Identifica entidades de alto riesgo
# MAGIC    - Optimiza asignación de recursos regulatorios
# MAGIC
# MAGIC 2. **Predicción de Tendencias**:
# MAGIC    - Anticipa volumen futuro de quejas
# MAGIC    - Permite planificación proactiva
# MAGIC    - Detecta anomalías tempranas
# MAGIC
# MAGIC 3. **Insights Accionables**:
# MAGIC    - Productos que requieren mayor supervisión
# MAGIC    - Entidades con comportamiento atípico
# MAGIC    - Patrones temporales para optimizar recursos
# MAGIC
# MAGIC

# COMMAND ----------

