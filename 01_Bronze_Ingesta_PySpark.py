# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Proyecto Final Big Data - Capa Bronze (PySpark)
# MAGIC %md
# MAGIC # 📊 PROYECTO FINAL BIG DATA - CAPA BRONZE (PySpark)
# MAGIC ## Ingesta de Datos de Quejas Financieras con Apache Spark
# MAGIC
# MAGIC ### 🎯 Objetivo
# MAGIC Este notebook implementa la **Capa Bronze** de la arquitectura Medallion usando **PySpark** para el análisis de quejas interpuestas por consumidores financieros ante la Superintendencia Financiera de Colombia.
# MAGIC
# MAGIC ### 📋 Contenido
# MAGIC 1. **Lectura de datos** desde tabla existente
# MAGIC 2. **Validación de calidad** con PySpark
# MAGIC 3. **Estadísticas descriptivas** usando DataFrame API
# MAGIC 4. **Persistencia** en formato Delta Lake
# MAGIC
# MAGIC ### 🏗️ Arquitectura Medallion
# MAGIC - **Bronze**: Datos crudos (raw) sin transformar
# MAGIC - **Silver**: Datos limpios y validados  
# MAGIC - **Gold**: Agregaciones y métricas de negocio
# MAGIC
# MAGIC ### 📅 Contexto del Proyecto
# MAGIC - **Dataset**: Quejas financieras Colombia (2015-2020)
# MAGIC - **Volumen**: ~150,000 registros
# MAGIC - **Entidades**: 463 instituciones financieras
# MAGIC - **Tecnología**: PySpark + Delta Lake + Unity Catalog
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Importar librerías y configuración
# Importar librerías necesarias
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime

# Configuración
print("✓ PySpark configurado correctamente")
print(f"Versión de Spark: {spark.version}")
print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# DBTITLE 1,Leer datos desde tabla Bronze existente
# Leer la tabla Bronze que ya fue creada
df_bronze = spark.table("workspace.bronze.quejas_financieras_raw")

print(f"✓ Tabla Bronze cargada")
print(f"Total de registros: {df_bronze.count():,}")
print(f"Total de columnas: {len(df_bronze.columns)}")

# COMMAND ----------

# DBTITLE 1,Explorar estructura de datos
# Ver esquema de la tabla
print("=" * 80)
print("ESQUEMA DE LA TABLA BRONZE")
print("=" * 80)
df_bronze.printSchema()

# Ver muestra de datos
print("\n" + "=" * 80)
print("MUESTRA DE DATOS (5 registros)")
print("=" * 80)
display(df_bronze.limit(5))

# COMMAND ----------

# DBTITLE 1,Estadísticas generales del dataset
# Calcular estadísticas generales usando PySpark
estadisticas = df_bronze.agg(
    F.count("*").alias("total_registros"),
    F.countDistinct("ANIO").alias("total_anios"),
    F.min("ANIO").alias("anio_inicial"),
    F.max("ANIO").alias("anio_final"),
    F.countDistinct("NOMBRE_ENTIDAD").alias("total_entidades"),
    F.countDistinct("PRODUCTO").alias("total_productos"),
    F.countDistinct("MOTIVO").alias("total_motivos"),
    F.countDistinct("ESTADO").alias("total_estados"),
    F.countDistinct("TIPO_ENTIDAD").alias("total_tipos_entidad")
)

print("\n" + "=" * 80)
print("ESTADÍSTICAS GENERALES DEL DATASET")
print("=" * 80)
display(estadisticas)

# COMMAND ----------

# DBTITLE 1,Validación de calidad - Valores nulos
# Análisis de valores nulos por columna usando PySpark
from pyspark.sql.functions import col, count, when, round as spark_round

total_registros = df_bronze.count()

# Calcular nulos para cada columna
nulos_info = []
for columna in df_bronze.columns:
    if columna != "_rescued_data":  # Excluir columna técnica
        nulos = df_bronze.select(
            F.lit(columna).alias("columna"),
            (F.count("*") - F.count(col(columna))).alias("valores_nulos"),
            spark_round(
                ((F.count("*") - F.count(col(columna))) * 100.0 / F.count("*")), 2
            ).alias("porcentaje_nulos")
        ).collect()[0]
        nulos_info.append(nulos)

# Convertir a DataFrame para visualización
df_nulos = spark.createDataFrame(nulos_info)

print("\n" + "=" * 80)
print("ANÁLISIS DE VALORES NULOS POR COLUMNA")
print("=" * 80)
display(df_nulos.orderBy(F.desc("porcentaje_nulos")))

# COMMAND ----------

# DBTITLE 1,Análisis de valores 'NA'
# Identificar registros con valores 'NA' en campos importantes
print("\n" + "=" * 80)
print("ANÁLISIS DE VALORES 'NA'")
print("=" * 80)

# Contar 'NA' en PRODUCTO
na_producto = df_bronze.filter(
    F.upper(F.trim(F.col("PRODUCTO"))) == "NA"
).count()

porcentaje_na_producto = round((na_producto * 100.0) / total_registros, 2)

print(f"\nPRODUCTO = 'NA':")
print(f"  Cantidad: {na_producto:,}")
print(f"  Porcentaje: {porcentaje_na_producto}%")

# Contar 'NA' en MOTIVO
na_motivo = df_bronze.filter(
    F.upper(F.trim(F.col("MOTIVO"))) == "NA"
).count()

porcentaje_na_motivo = round((na_motivo * 100.0) / total_registros, 2)

print(f"\nMOTIVO = 'NA':")
print(f"  Cantidad: {na_motivo:,}")
print(f"  Porcentaje: {porcentaje_na_motivo}%")

# COMMAND ----------

# DBTITLE 1,Distribución temporal de quejas
# Distribución de quejas por año usando PySpark
df_temporal = df_bronze.groupBy(
    F.col("ANIO").cast("int").alias("anio")
).agg(
    F.count("*").alias("total_quejas"),
    F.countDistinct("NOMBRE_ENTIDAD").alias("entidades_con_quejas"),
    F.round(F.avg("MES"), 1).alias("mes_promedio")
).orderBy("anio")

print("\n" + "=" * 80)
print("DISTRIBUCIÓN TEMPORAL DE QUEJAS POR AÑO")
print("=" * 80)
display(df_temporal)

# COMMAND ----------

# DBTITLE 1,Top 10 entidades con más quejas
# Top 10 entidades con mayor número de quejas
df_top_entidades = df_bronze.groupBy("NOMBRE_ENTIDAD").agg(
    F.count("*").alias("total_quejas"),
    F.countDistinct("PRODUCTO").alias("productos_diferentes"),
    F.countDistinct("MOTIVO").alias("motivos_diferentes")
).orderBy(F.desc("total_quejas")).limit(10)

print("\n" + "=" * 80)
print("TOP 10 ENTIDADES FINANCIERAS CON MÁS QUEJAS")
print("=" * 80)
display(df_top_entidades)

# COMMAND ----------

# DBTITLE 1,Distribución por tipo de entidad
# Distribución de quejas por tipo de entidad
df_tipos = df_bronze.groupBy("TIPO_ENTIDAD").agg(
    F.count("*").alias("total_quejas"),
    F.countDistinct("NOMBRE_ENTIDAD").alias("cantidad_entidades")
).orderBy(F.desc("total_quejas"))

# Agregar porcentaje
df_tipos = df_tipos.withColumn(
    "porcentaje",
    F.round((F.col("total_quejas") * 100.0 / total_registros), 2)
)

print("\n" + "=" * 80)
print("DISTRIBUCIÓN POR TIPO DE ENTIDAD")
print("=" * 80)
display(df_tipos)

# COMMAND ----------

# DBTITLE 1,Resumen y próximos pasos
# MAGIC %md
# MAGIC ## ✅ Resumen Capa Bronze (PySpark)
# MAGIC
# MAGIC ### Datos cargados exitosamente:
# MAGIC - ✓ Tabla: `workspace.bronze.quejas_financieras_raw`
# MAGIC - ✓ Registros: ~150,000 quejas
# MAGIC - ✓ Periodo: 2015-2020
# MAGIC - ✓ Entidades: 463 instituciones financieras
# MAGIC - ✓ Productos: 65 tipos diferentes
# MAGIC - ✓ Tecnología: PySpark + Delta Lake
# MAGIC
# MAGIC ### Hallazgos de calidad identificados:
# MAGIC - ✓ Presencia de valores 'NA' en campos PRODUCTO y MOTIVO
# MAGIC - ✓ Necesidad de normalizar formatos de año (2.016 → 2016)
# MAGIC - ✓ Oportunidad de enriquecimiento con categorías
# MAGIC - ✓ Datos listos para transformación en capa Silver
# MAGIC
# MAGIC ### 🔜 Próximos pasos:
# MAGIC 1. **Capa Silver (PySpark)**: Limpieza y transformación
# MAGIC    - Normalización de formatos
# MAGIC    - Tratamiento de valores 'NA'
# MAGIC    - Creación de campos de fecha
# MAGIC    - Categorización de tipos de entidad
# MAGIC
# MAGIC 2. **Capa Gold (PySpark)**: Agregaciones y métricas
# MAGIC    - KPIs de negocio
# MAGIC    - Métricas por entidad, producto y tiempo
# MAGIC    - Tablas analíticas optimizadas
# MAGIC
# MAGIC 3. **Análisis y ML**: Modelos predictivos
# MAGIC    - Clustering de entidades
# MAGIC    - Forecasting de quejas
# MAGIC    - Análisis de patrones

# COMMAND ----------

