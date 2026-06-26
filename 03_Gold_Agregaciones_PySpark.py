# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Proyecto Final Big Data - Capa Gold (PySpark)
# MAGIC %md
# MAGIC # 🥇 PROYECTO FINAL BIG DATA - CAPA GOLD (PySpark)
# MAGIC ## Agregaciones y Métricas de Negocio con Apache Spark
# MAGIC
# MAGIC ### 🎯 Objetivo
# MAGIC Este notebook implementa la **Capa Gold** usando **PySpark**, creando tablas analíticas con agregaciones y métricas clave para el negocio.
# MAGIC
# MAGIC ### 📊 Tablas Gold a crear:
# MAGIC 1. **gold.metricas_por_entidad**: KPIs por entidad financiera
# MAGIC 2. **gold.metricas_por_producto**: Análisis por producto financiero
# MAGIC 3. **gold.metricas_temporales**: Evolución mensual y tendencias
# MAGIC 4. **gold.ranking_entidades**: Top entidades por diversos criterios
# MAGIC 5. **gold.dashboard_kpis**: Métricas principales para dashboards
# MAGIC
# MAGIC ### 🛠️ Técnicas PySpark utilizadas:
# MAGIC - `groupBy()` y `agg()` para agregaciones
# MAGIC - `Window functions` para rankings y tendencias
# MAGIC - `pivot()` para matrices de datos
# MAGIC - Delta Lake para almacenamiento optimizado
# MAGIC
# MAGIC ### 💼 Valor de negocio:
# MAGIC - Identificar entidades con más quejas
# MAGIC - Detectar productos problemáticos
# MAGIC - Monitorear tendencias temporales
# MAGIC - Apoyar toma de decisiones regulatorias
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Importar librerías y configuración
# Importar librerías necesarias
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime

print("✓ Librerías importadas correctamente")
print(f"Versión de Spark: {spark.version}")
print(f"Inicio del procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# DBTITLE 1,Leer datos de la capa Silver
# Leer tabla Silver
df_silver = spark.table("workspace.silver.quejas_limpias")

print(f"✓ Tabla Silver cargada")
print(f"Registros totales: {df_silver.count():,}")
print(f"Periodo: {df_silver.select(F.min('anio')).collect()[0][0]} - {df_silver.select(F.max('anio')).collect()[0][0]}")

# COMMAND ----------

# DBTITLE 1,Tabla Gold 1: Métricas por Entidad
# Crear tabla Gold con métricas por entidad financiera
df_metricas_entidad = df_silver.groupBy(
    "codigo_entidad",
    "nombre_entidad",
    "tipo_entidad_desc"
).agg(
    # Métricas de volumen
    F.count("*").alias("total_quejas"),
    F.countDistinct("producto").alias("productos_distintos"),
    F.countDistinct("motivo").alias("motivos_distintos"),
    
    # Métricas temporales
    F.min("fecha_queja").alias("primera_queja"),
    F.max("fecha_queja").alias("ultima_queja"),
    F.countDistinct("periodo_mes").alias("meses_con_quejas"),
    
    # Métricas por estado
    F.sum(F.when(F.col("estado") == "VIGENTES", 1).otherwise(0)).alias("quejas_vigentes"),
    F.sum(F.when(F.col("estado") == "RECIBIDOS", 1).otherwise(0)).alias("quejas_recibidas"),
    
    # Promedios
    F.round(F.count("*") / F.countDistinct("periodo_mes"), 2).alias("promedio_quejas_por_mes")
)

# Agregar ranking
window_ranking = Window.orderBy(F.desc("total_quejas"))
df_metricas_entidad = df_metricas_entidad.withColumn(
    "ranking_quejas",
    F.row_number().over(window_ranking)
)

# Guardar en Delta Lake
df_metricas_entidad.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.metricas_por_entidad")

print("✓ Tabla Gold: metricas_por_entidad creada")
print(f"Entidades analizadas: {df_metricas_entidad.count()}")

# Mostrar top 10
print("\nTop 10 entidades con más quejas:")
display(df_metricas_entidad.orderBy("ranking_quejas").limit(10))

# COMMAND ----------

# DBTITLE 1,Tabla Gold 2: Métricas por Producto
# Crear tabla Gold con métricas por producto financiero
df_metricas_producto = df_silver.groupBy(
    "producto"
).agg(
    # Métricas de volumen
    F.count("*").alias("total_quejas"),
    F.countDistinct("nombre_entidad").alias("entidades_afectadas"),
    F.countDistinct("motivo").alias("motivos_distintos"),
    
    # Distribución por tipo de entidad
    F.sum(F.when(F.col("tipo_entidad_desc") == "BANCO", 1).otherwise(0)).alias("quejas_bancos"),
    F.sum(F.when(F.col("tipo_entidad_desc") == "PENSION Y CESANTIAS", 1).otherwise(0)).alias("quejas_pensiones"),
    F.sum(F.when(F.col("tipo_entidad_desc") == "SEGUROS", 1).otherwise(0)).alias("quejas_seguros"),
    
    # Métricas temporales
    F.min("anio").alias("anio_inicial"),
    F.max("anio").alias("anio_final"),
    F.countDistinct("periodo_mes").alias("meses_con_quejas"),
    
    # Porcentaje y ranking
    F.round(
        F.count("*") * 100.0 / df_silver.count(), 2
    ).alias("porcentaje_total")
)

# Agregar ranking
window_ranking = Window.orderBy(F.desc("total_quejas"))
df_metricas_producto = df_metricas_producto.withColumn(
    "ranking_producto",
    F.row_number().over(window_ranking)
)

# Guardar en Delta Lake
df_metricas_producto.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.metricas_por_producto")

print("✓ Tabla Gold: metricas_por_producto creada")
print(f"Productos analizados: {df_metricas_producto.count()}")

# Mostrar top 15
print("\nTop 15 productos con más quejas:")
display(df_metricas_producto.orderBy("ranking_producto").limit(15))

# COMMAND ----------

# DBTITLE 1,Tabla Gold 3: Métricas Temporales
# Crear tabla Gold con evolución temporal mensual
df_metricas_temporales = df_silver.groupBy(
    "anio",
    "mes",
    "periodo_mes",
    "fecha_queja"
).agg(
    # Métricas de volumen
    F.count("*").alias("total_quejas"),
    F.countDistinct("nombre_entidad").alias("entidades_con_quejas"),
    F.countDistinct("producto").alias("productos_distintos"),
    
    # Distribución por tipo de entidad
    F.sum(F.when(F.col("tipo_entidad_desc") == "BANCO", 1).otherwise(0)).alias("quejas_bancos"),
    F.sum(F.when(F.col("tipo_entidad_desc") == "PENSION Y CESANTIAS", 1).otherwise(0)).alias("quejas_pensiones"),
    F.sum(F.when(F.col("tipo_entidad_desc") == "SEGUROS", 1).otherwise(0)).alias("quejas_seguros"),
    
    # Distribución por estado
    F.sum(F.when(F.col("estado") == "VIGENTES", 1).otherwise(0)).alias("quejas_vigentes"),
    F.sum(F.when(F.col("estado") == "RECIBIDOS", 1).otherwise(0)).alias("quejas_recibidas")
).orderBy("fecha_queja")

# Agregar métricas de tendencia usando Window functions
window_spec = Window.orderBy("fecha_queja").rowsBetween(-2, 0)

df_metricas_temporales = df_metricas_temporales.withColumn(
    "promedio_movil_3meses",
    F.round(F.avg("total_quejas").over(window_spec), 2)
).withColumn(
    "tendencia",
    F.when(
        F.col("total_quejas") > F.lag("total_quejas", 1).over(Window.orderBy("fecha_queja")),
        "AUMENTO"
    ).when(
        F.col("total_quejas") < F.lag("total_quejas", 1).over(Window.orderBy("fecha_queja")),
        "DISMINUCION"
    ).otherwise("ESTABLE")
)

# Guardar en Delta Lake
df_metricas_temporales.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.metricas_temporales")

print("✓ Tabla Gold: metricas_temporales creada")
print(f"Periodos mensuales analizados: {df_metricas_temporales.count()}")

# Mostrar últimos 12 meses
print("\nÚltimos 12 meses:")
display(df_metricas_temporales.orderBy(F.desc("fecha_queja")).limit(12))

# COMMAND ----------

# DBTITLE 1,Tabla Gold 4: Ranking de Entidades
# Crear tabla Gold con rankings de entidades
# Ranking por total de quejas
df_ranking_total = df_silver.groupBy(
    "nombre_entidad",
    "tipo_entidad_desc"
).agg(
    F.count("*").alias("total_quejas")
).withColumn(
    "ranking_tipo",
    F.lit("total_quejas")
).withColumn(
    "posicion",
    F.row_number().over(Window.orderBy(F.desc("total_quejas")))
)

# Ranking por diversidad de productos (entidades con más productos problemáticos)
df_ranking_productos = df_silver.groupBy(
    "nombre_entidad",
    "tipo_entidad_desc"
).agg(
    F.countDistinct("producto").alias("total_quejas")
).withColumn(
    "ranking_tipo",
    F.lit("productos_distintos")
).withColumn(
    "posicion",
    F.row_number().over(Window.orderBy(F.desc("total_quejas")))
)

# Ranking por quejas recientes (2020)
df_ranking_reciente = df_silver.filter(
    F.col("anio") == 2020
).groupBy(
    "nombre_entidad",
    "tipo_entidad_desc"
).agg(
    F.count("*").alias("total_quejas")
).withColumn(
    "ranking_tipo",
    F.lit("quejas_2020")
).withColumn(
    "posicion",
    F.row_number().over(Window.orderBy(F.desc("total_quejas")))
)

# Unir todos los rankings
df_ranking_entidades = df_ranking_total.union(
    df_ranking_productos
).union(
    df_ranking_reciente
)

# Guardar en Delta Lake
df_ranking_entidades.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.ranking_entidades")

print("✓ Tabla Gold: ranking_entidades creada")

# Mostrar top 10 por cada tipo de ranking
print("\nTop 10 por total de quejas:")
display(
    df_ranking_entidades
    .filter(F.col("ranking_tipo") == "total_quejas")
    .orderBy("posicion")
    .limit(10)
)

# COMMAND ----------

# DBTITLE 1,Tabla Gold 5: KPIs para Dashboard
# Crear tabla Gold con KPIs principales para dashboards
total_quejas = df_silver.count()

df_dashboard_kpis = spark.createDataFrame([
    {
        "kpi_nombre": "Total Quejas",
        "kpi_valor": float(total_quejas),
        "kpi_categoria": "GENERAL",
        "kpi_descripcion": "Total de quejas registradas en el periodo 2015-2020"
    },
    {
        "kpi_nombre": "Entidades con Quejas",
        "kpi_valor": float(df_silver.select(F.countDistinct("nombre_entidad")).collect()[0][0]),
        "kpi_categoria": "GENERAL",
        "kpi_descripcion": "Número de entidades financieras con al menos una queja"
    },
    {
        "kpi_nombre": "Promedio Quejas por Entidad",
        "kpi_valor": round(total_quejas / df_silver.select(F.countDistinct("nombre_entidad")).collect()[0][0], 2),
        "kpi_categoria": "PROMEDIO",
        "kpi_descripcion": "Promedio de quejas por entidad financiera"
    },
    {
        "kpi_nombre": "Porcentaje Quejas Bancos",
        "kpi_valor": round(
            df_silver.filter(F.col("tipo_entidad_desc") == "BANCO").count() * 100.0 / total_quejas, 2
        ),
        "kpi_categoria": "DISTRIBUCION",
        "kpi_descripcion": "Porcentaje de quejas contra bancos"
    },
    {
        "kpi_nombre": "Productos Distintos",
        "kpi_valor": float(df_silver.select(F.countDistinct("producto")).collect()[0][0]),
        "kpi_categoria": "GENERAL",
        "kpi_descripcion": "Número de productos financieros con quejas"
    },
    {
        "kpi_nombre": "Quejas Vigentes Porcentaje",
        "kpi_valor": round(
            df_silver.filter(F.col("estado") == "VIGENTES").count() * 100.0 / total_quejas, 2
        ),
        "kpi_categoria": "ESTADO",
        "kpi_descripcion": "Porcentaje de quejas con estado VIGENTES"
    }
])

# Agregar timestamp
df_dashboard_kpis = df_dashboard_kpis.withColumn(
    "fecha_actualizacion",
    F.current_timestamp()
)

# Guardar en Delta Lake
df_dashboard_kpis.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.dashboard_kpis")

print("✓ Tabla Gold: dashboard_kpis creada")
print("\nKPIs principales:")
display(df_dashboard_kpis)

# COMMAND ----------

# DBTITLE 1,Resumen de tablas Gold creadas
# Resumen de todas las tablas Gold creadas
print("=" * 80)
print("RESUMEN DE TABLAS GOLD CREADAS")
print("=" * 80)

tablas_gold = [
    "workspace.gold.metricas_por_entidad",
    "workspace.gold.metricas_por_producto",
    "workspace.gold.metricas_temporales",
    "workspace.gold.ranking_entidades",
    "workspace.gold.dashboard_kpis"
]

for tabla in tablas_gold:
    count = spark.table(tabla).count()
    print(f"\n✓ {tabla}")
    print(f"  Registros: {count:,}")

print("\n" + "=" * 80)
print("TODAS LAS TABLAS GOLD CREADAS EXITOSAMENTE")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Resumen y próximos pasos
# MAGIC %md
# MAGIC ## ✅ Resumen Capa Gold (PySpark)
# MAGIC
# MAGIC ### Tablas Gold creadas exitosamente:
# MAGIC 1. ✓ **gold.metricas_por_entidad**: Métricas completas por institución financiera
# MAGIC    - Total quejas, productos, motivos
# MAGIC    - Ranking de entidades
# MAGIC    - Promedio mensual de quejas
# MAGIC
# MAGIC 2. ✓ **gold.metricas_por_producto**: Análisis por producto financiero
# MAGIC    - Volumen de quejas por producto
# MAGIC    - Entidades afectadas
# MAGIC    - Distribución por tipo de entidad
# MAGIC
# MAGIC 3. ✓ **gold.metricas_temporales**: Evolución temporal
# MAGIC    - Serie mensual completa 2015-2020
# MAGIC    - Promedio móvil de 3 meses
# MAGIC    - Tendencias (aumento/disminución)
# MAGIC
# MAGIC 4. ✓ **gold.ranking_entidades**: Rankings múltiples
# MAGIC    - Por total de quejas
# MAGIC    - Por diversidad de productos
# MAGIC    - Por quejas recientes (2020)
# MAGIC
# MAGIC 5. ✓ **gold.dashboard_kpis**: KPIs principales
# MAGIC    - Métricas generales
# MAGIC    - Porcentajes clave
# MAGIC    - Promedios de negocio
# MAGIC
# MAGIC ### Técnicas PySpark aplicadas:
# MAGIC - ✓ Agregaciones complejas con `groupBy()` y `agg()`
# MAGIC - ✓ Window functions para rankings y promedios móviles
# MAGIC - ✓ Expresiones condicionales con `when().otherwise()`
# MAGIC - ✓ Delta Lake para almacenamiento optimizado
# MAGIC - ✓ Particionamiento lógico por entidad y tiempo
# MAGIC
# MAGIC ### Valor de negocio generado:
# MAGIC - 📊 Identificación de entidades con mayor volumen de quejas
# MAGIC - 🔍 Detección de productos problemáticos
# MAGIC - 📈 Monitoreo de tendencias temporales
# MAGIC - 🎯 Apoyo a decisiones regulatorias y supervisión
# MAGIC - 📊 Base para dashboards interactivos
# MAGIC
# MAGIC ### 🔜 Próximos pasos:
# MAGIC 1. **Análisis Avanzado y Machine Learning**:
# MAGIC    - Clustering de entidades similares
# MAGIC    - Forecasting de quejas futuras
# MAGIC    - Detección de anomalías
# MAGIC    - Análisis de texto de motivos
# MAGIC
# MAGIC 2. **Dashboards y Visualización**:
# MAGIC    - Dashboard interactivo con gráficos
# MAGIC    - KPIs dinámicos
# MAGIC    - Tendencias temporales visuales
# MAGIC
# MAGIC 3. **Automatización**:
# MAGIC    - Pipeline DLT (Lakeflow Spark Declarative)
# MAGIC    - Scheduled Jobs para actualización periódica
# MAGIC    - Alertas automáticas

# COMMAND ----------

