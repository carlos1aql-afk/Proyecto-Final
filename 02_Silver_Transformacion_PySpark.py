# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Proyecto Final Big Data - Capa Silver (PySpark)
# MAGIC %md
# MAGIC # ✨ PROYECTO FINAL BIG DATA - CAPA SILVER (PySpark)
# MAGIC ## Transformación y Limpieza de Datos con Apache Spark
# MAGIC
# MAGIC ### 🎯 Objetivo
# MAGIC Este notebook implementa la **Capa Silver** usando **PySpark DataFrame API**, aplicando transformaciones y limpieza sobre los datos crudos de la capa Bronze.
# MAGIC
# MAGIC ### 🔧 Transformaciones aplicadas con PySpark:
# MAGIC 1. **Normalización de formato de año** (2.016 → 2016)
# MAGIC 2. **Creación de fecha completa** usando `to_date()`
# MAGIC 3. **Tratamiento de valores 'NA'** con expresiones `when()`
# MAGIC 4. **Categorización de entidades** por tipo usando `case when`
# MAGIC 5. **Limpieza de texto** con funciones `trim()`, `upper()`
# MAGIC 6. **Enriquecimiento** con campos calculados
# MAGIC 7. **Persistencia en Delta Lake** con formato optimizado
# MAGIC
# MAGIC ### 📊 Resultado:
# MAGIC Tabla `workspace.silver.quejas_limpias` lista para análisis de negocio
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Importar librerías PySpark
# Importar librerías necesarias
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime

print("✓ Librerías importadas correctamente")
print(f"Versión de Spark: {spark.version}")
print(f"Inicio del procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# DBTITLE 1,Leer datos de la capa Bronze
# Leer tabla Bronze
df_bronze = spark.table("workspace.bronze.quejas_financieras_raw")

print(f"✓ Tabla Bronze cargada")
print(f"Registros totales: {df_bronze.count():,}")
print(f"\nColumnas: {df_bronze.columns}")

# COMMAND ----------

# DBTITLE 1,Aplicar transformaciones y crear tabla Silver
# Crear DataFrame Silver con todas las transformaciones usando PySpark
df_silver = df_bronze \
    .filter(
        # Filtrar registros válidos (excluir 'NA' antes de cast)
        (F.col("ANIO").isNotNull()) & 
        (F.col("MES").isNotNull()) & 
        (F.col("MES").between(1, 12)) &
        (F.upper(F.trim(F.col("TIPO_ENTIDAD"))) != "NA") &
        (F.upper(F.trim(F.col("CODIGO_ENTIDAD"))) != "NA")
    ) \
    .select(
        # IDs y clasificación
        F.col("TIPO_ENTIDAD").cast("int").alias("tipo_entidad_id"),
        F.col("CODIGO_ENTIDAD").cast("int").alias("codigo_entidad"),
        
        # Normalización de nombre de entidad (UPPER + eliminar sufijos legales + TRIM)
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.upper(F.col("NOMBRE_ENTIDAD")),
                    r'\s+S\.A\.?\s*$', ''
                ),
                r'\s+LTDA\.?\s*$', ''
            )
        ).alias("nombre_entidad"),
        
        # Producto (tratamiento de NA)
        F.when(
            (F.upper(F.trim(F.col("PRODUCTO"))) == "NA") | F.col("PRODUCTO").isNull(),
            "NO ESPECIFICADO"
        ).otherwise(
            F.trim(F.col("PRODUCTO"))
        ).alias("producto"),
        
        # Motivo (tratamiento de NA)
        F.when(
            (F.upper(F.trim(F.col("MOTIVO"))) == "NA") | F.col("MOTIVO").isNull(),
            "NO ESPECIFICADO"
        ).otherwise(
            F.trim(F.col("MOTIVO"))
        ).alias("motivo"),
        
        # Estado
        F.trim(F.upper(F.col("ESTADO"))).alias("estado"),
        
        # Normalización de fecha
        F.col("ANIO").cast("int").alias("anio"),
        F.col("MES").cast("int").alias("mes"),
        
        # Crear fecha completa (primer día del mes)
        F.make_date(
            F.col("ANIO").cast("int"),
            F.col("MES").cast("int"),
            F.lit(1)
        ).alias("fecha_queja"),
        
        # Campos calculados
        F.concat(
            F.col("ANIO").cast("int").cast("string"),
            F.lit("-"),
            F.lpad(F.col("MES").cast("int").cast("string"), 2, "0")
        ).alias("periodo_mes"),
        
        F.concat(
            F.lit("Q"),
            F.quarter(
                F.make_date(
                    F.col("ANIO").cast("int"),
                    F.col("MES").cast("int"),
                    F.lit(1)
                )
            ).cast("string")
        ).alias("trimestre"),
        
        # Categorización de tipo de entidad usando case when
        F.when(F.col("TIPO_ENTIDAD").cast("int") == 1, "BANCO")
         .when(F.col("TIPO_ENTIDAD").cast("int").isin([23, 24, 25]), "PENSION Y CESANTIAS")
         .when(F.col("TIPO_ENTIDAD").cast("int").isin([2, 3]), "CORPORACION FINANCIERA")
         .when(F.col("TIPO_ENTIDAD").cast("int").isin([5, 6]), "SEGUROS")
         .when(F.col("TIPO_ENTIDAD").cast("int").isin([7, 8]), "SOCIEDADES FIDUCIARIAS")
         .otherwise("OTRA ENTIDAD FINANCIERA")
         .alias("tipo_entidad_desc"),
        
        # Timestamp de procesamiento
        F.current_timestamp().alias("fecha_procesamiento")
    )

print(f"✓ Transformaciones aplicadas")
print(f"Registros después de transformación: {df_silver.count():,}")

# COMMAND ----------

# DBTITLE 1,Guardar tabla Silver en Delta Lake
# Escribir tabla Silver en formato Delta Lake
df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.silver.quejas_limpias")

print("✓ Tabla Silver creada exitosamente")
print("Ubicación: workspace.silver.quejas_limpias")
print("Formato: Delta Lake")

# COMMAND ----------

# DBTITLE 1,Verificar tabla Silver creada
# Leer y verificar la tabla Silver creada
df_silver_verify = spark.table("workspace.silver.quejas_limpias")

# Estadísticas de verificación
verificacion = df_silver_verify.agg(
    F.count("*").alias("total_registros"),
    F.countDistinct("anio").alias("total_anios"),
    F.min("fecha_queja").alias("fecha_minima"),
    F.max("fecha_queja").alias("fecha_maxima"),
    F.countDistinct("nombre_entidad").alias("total_entidades"),
    F.countDistinct("producto").alias("total_productos")
)

print("\n" + "=" * 80)
print("VERIFICACIÓN DE TABLA SILVER")
print("=" * 80)
display(verificacion)

# COMMAND ----------

# DBTITLE 1,Muestra de datos transformados
# Ver muestra de los datos transformados
print("\n" + "=" * 80)
print("MUESTRA DE DATOS TRANSFORMADOS (10 registros)")
print("=" * 80)

df_muestra = df_silver_verify.select(
    "tipo_entidad_desc",
    "nombre_entidad",
    "producto",
    "motivo",
    "estado",
    "fecha_queja",
    "periodo_mes",
    "trimestre"
).limit(10)

display(df_muestra)

# COMMAND ----------

# DBTITLE 1,Validación - Valores 'NA' tratados
# Verificar que los valores 'NA' fueron tratados correctamente
total = df_silver_verify.count()

# Producto NO ESPECIFICADO
no_esp_producto = df_silver_verify.filter(
    F.col("producto") == "NO ESPECIFICADO"
).count()

# Motivo NO ESPECIFICADO
no_esp_motivo = df_silver_verify.filter(
    F.col("motivo") == "NO ESPECIFICADO"
).count()

print("\n" + "=" * 80)
print("VALIDACIÓN - VALORES 'NA' TRATADOS")
print("=" * 80)
print(f"\nProducto 'NO ESPECIFICADO':")
print(f"  Cantidad: {no_esp_producto:,}")
print(f"  Porcentaje: {round(no_esp_producto * 100.0 / total, 2)}%")

print(f"\nMotivo 'NO ESPECIFICADO':")
print(f"  Cantidad: {no_esp_motivo:,}")
print(f"  Porcentaje: {round(no_esp_motivo * 100.0 / total, 2)}%")

# COMMAND ----------

# DBTITLE 1,Distribución por tipo de entidad
# Distribución de quejas por tipo de entidad (categorizado)
df_distribucion = df_silver_verify.groupBy("tipo_entidad_desc").agg(
    F.count("*").alias("total_quejas"),
    F.countDistinct("nombre_entidad").alias("cantidad_entidades")
).withColumn(
    "porcentaje_total",
    F.round((F.col("total_quejas") * 100.0 / total), 2)
).orderBy(F.desc("total_quejas"))

print("\n" + "=" * 80)
print("DISTRIBUCIÓN POR TIPO DE ENTIDAD")
print("=" * 80)
display(df_distribucion)

# COMMAND ----------

# DBTITLE 1,Validación de fechas por año
# Validar que las fechas fueron creadas correctamente
df_validacion_fechas = df_silver_verify.groupBy("anio").agg(
    F.count("*").alias("registros"),
    F.min("fecha_queja").alias("primera_fecha"),
    F.max("fecha_queja").alias("ultima_fecha"),
    F.countDistinct("periodo_mes").alias("meses_con_datos")
).orderBy("anio")

print("\n" + "=" * 80)
print("VALIDACIÓN DE FECHAS POR AÑO")
print("=" * 80)
display(df_validacion_fechas)

# COMMAND ----------

# DBTITLE 1,Análisis de completitud por columna
# Verificar completitud de datos en tabla Silver
completitud = df_silver_verify.agg(
    F.count("*").alias("total_registros"),
    F.count("nombre_entidad").alias("nombre_entidad_completo"),
    F.count("producto").alias("producto_completo"),
    F.count("motivo").alias("motivo_completo"),
    F.count("estado").alias("estado_completo"),
    F.count("fecha_queja").alias("fecha_queja_completa")
).withColumn(
    "porcentaje_completitud",
    F.round((F.col("fecha_queja_completa") * 100.0 / F.col("total_registros")), 2)
)

print("\n" + "=" * 80)
print("ANÁLISIS DE COMPLETITUD POR COLUMNA")
print("=" * 80)
display(completitud)

# COMMAND ----------

# DBTITLE 1,Resumen y próximos pasos
# MAGIC %md
# MAGIC ## ✅ Resumen Capa Silver (PySpark)
# MAGIC
# MAGIC ### Transformaciones aplicadas exitosamente con PySpark:
# MAGIC - ✓ Normalización de formato de año y mes usando `cast()`
# MAGIC - ✓ Creación de campo `fecha_queja` con `make_date()`
# MAGIC - ✓ Tratamiento de valores 'NA' con `when().otherwise()`
# MAGIC - ✓ Categorización de tipos de entidad con `case when`
# MAGIC - ✓ Campos calculados: periodo_mes, trimestre con funciones PySpark
# MAGIC - ✓ Limpieza y estandarización con `trim()`, `upper()`
# MAGIC - ✓ Persistencia en Delta Lake con `saveAsTable()`
# MAGIC
# MAGIC ### Calidad de datos mejorada:
# MAGIC - Sin valores nulos críticos
# MAGIC - Fechas validadas y consistentes
# MAGIC - Categorías estandarizadas
# MAGIC - Datos listos para análisis de negocio
# MAGIC - Formato Delta Lake optimizado
# MAGIC
# MAGIC ### Ventajas de usar PySpark:
# MAGIC - ✓ Procesamiento distribuido y escalable
# MAGIC - ✓ Optimización automática de queries (Catalyst)
# MAGIC - ✓ Operaciones lazy evaluation
# MAGIC - ✓ Soporte nativo para Delta Lake
# MAGIC - ✓ API declarativa y expresiva
# MAGIC
# MAGIC ### 🔜 Próximos pasos:
# MAGIC 1. **Capa Gold (PySpark)**: Crear agregaciones y métricas de negocio
# MAGIC 2. **KPIs**: Calcular indicadores clave con agregaciones PySpark
# MAGIC 3. **Análisis temporal**: Tendencias usando Window functions
# MAGIC 4. **Segmentación**: Agrupaciones con `groupBy()` y `agg()`

# COMMAND ----------

