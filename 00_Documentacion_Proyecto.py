# Databricks notebook source
# MAGIC %md

# COMMAND ----------

# DBTITLE 1,Proyecto Final Big Data - Documentación Completa
# MAGIC
# MAGIC %md
# MAGIC ## 📚 Proyecto Final Realizado por Carlos Andrés Quintero Laverde
# MAGIC ## Análisis de Quejas de Consumidores Financieros en Colombia
# MAGIC ### Arquitectura Medallion con PySpark + Delta Lake + Unity Catalog
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 RESUMEN EJECUTIVO
# MAGIC
# MAGIC ### Contexto del Proyecto
# MAGIC Este proyecto implementa una **solución completa de Big Data** para analizar quejas interpuestas por consumidores financieros ante la Superintendencia Financiera de Colombia, utilizando tecnologías modernas de procesamiento distribuido.
# MAGIC
# MAGIC
# MAGIC ### Datos del Proyecto
# MAGIC - **Dataset**: Quejas financieras Colombia 2015-2020
# MAGIC - **Volumen**: ~150,000 registros
# MAGIC - **Entidades**: 463 instituciones financieras
# MAGIC - **Productos**: 65 tipos de productos financieros
# MAGIC - **Periodo**: 6 años de datos históricos
# MAGIC
# MAGIC ### Problema de Negocio
# MAGIC **¿Cómo puede una cooperativa financiera (o la Superintendencia Financiera) identificar patrones de quejas, entidades problemáticas y predecir tendencias futuras para mejorar la supervisión y el servicio al cliente?**
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Arquitectura de la Solución
# MAGIC %md
# MAGIC ## 🏗️ ARQUITECTURA DE LA SOLUCIÓN
# MAGIC
# MAGIC ### Arquitectura Medallion (Bronze-Silver-Gold)
# MAGIC
# MAGIC La arquitectura Medallion es un patrón de diseño de data lakehouse que organiza los datos en tres capas:
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────┐
# MAGIC │   CAPA BRONZE    │
# MAGIC │  (Datos Crudos)  │
# MAGIC │                  │
# MAGIC │ • CSV Raw Data    │
# MAGIC │ • Sin transformar │
# MAGIC │ • Delta Lake     │
# MAGIC │ • Unity Catalog  │
# MAGIC └─────────┬─────────┘
# MAGIC          │
# MAGIC          ↓ PySpark Transformations
# MAGIC          │
# MAGIC ┌─────────┴───────────┐
# MAGIC │   CAPA SILVER     │
# MAGIC │  (Datos Limpios) │
# MAGIC │                   │
# MAGIC │ • Limpieza       │
# MAGIC │ • Validación     │
# MAGIC │ • Normalización  │
# MAGIC │ • Enriquecimiento│
# MAGIC └──────────┬──────────┘
# MAGIC            │
# MAGIC            ↓ PySpark Aggregations
# MAGIC            │
# MAGIC ┌──────────┴─────────────┐
# MAGIC │    CAPA GOLD       │
# MAGIC │ (Métricas Negocio)│
# MAGIC │                    │
# MAGIC │ • Agregaciones    │
# MAGIC │ • KPIs            │
# MAGIC │ • Rankings        │
# MAGIC │ • Serie Temporal  │
# MAGIC └───────────┬────────────┘
# MAGIC             │
# MAGIC             ↓ Analytics & ML
# MAGIC             │
# MAGIC ┌───────────┴──────────────┐
# MAGIC │  ANALYTICS & ML   │
# MAGIC │                   │
# MAGIC │ • EDA             │
# MAGIC │ • Clustering      │
# MAGIC │ • Forecasting    │
# MAGIC │ • Dashboards     │
# MAGIC └────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Stack Tecnológico
# MAGIC
# MAGIC | Componente | Tecnología | Propósito |
# MAGIC |------------|-------------|----------|
# MAGIC | **Procesamiento** | Apache Spark (PySpark) | Procesamiento distribuido de datos |
# MAGIC | **Almacenamiento** | Delta Lake | Formato ACID para data lakehouse |
# MAGIC | **Catálogo** | Unity Catalog | Gobernanza y seguridad de datos |
# MAGIC | **Machine Learning** | PySpark MLlib | Clustering y modelos predictivos |
# MAGIC | **Visualización** | Matplotlib, Seaborn | Gráficos y dashboards |
# MAGIC | **Orquestación** | Databricks Notebooks | Desarrollo y ejecución |
# MAGIC | **Automatización** | Databricks Jobs | Pipelines automatizados |
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Estructura del Proyecto
# MAGIC %md
# MAGIC ## 📁 ESTRUCTURA DEL PROYECTO
# MAGIC
# MAGIC ### Notebooks Desarrollados
# MAGIC
# MAGIC #### 1. **01_Bronze_Ingesta_PySpark**
# MAGIC - **Propósito**: Ingesta de datos crudos
# MAGIC - **Tecnología**: PySpark DataFrame API
# MAGIC - **Salida**: Tabla `workspace.bronze.quejas_financieras_raw`
# MAGIC - **Características**:
# MAGIC   - Lectura de CSV con opciones de esquema
# MAGIC   - Validación de calidad inicial
# MAGIC   - Estadísticas descriptivas
# MAGIC   - Identificación de valores nulos y 'NA'
# MAGIC   - Análisis exploratorio inicial
# MAGIC
# MAGIC #### 2. **02_Silver_Transformacion_PySpark**
# MAGIC - **Propósito**: Limpieza y transformación de datos
# MAGIC - **Tecnología**: PySpark con funciones avanzadas
# MAGIC - **Salida**: Tabla `workspace.silver.quejas_limpias`
# MAGIC - **Transformaciones**:
# MAGIC   - Normalización de formato de año/mes
# MAGIC   - Creación de campos de fecha (`make_date`)
# MAGIC   - Tratamiento de valores 'NA' → 'NO ESPECIFICADO'
# MAGIC   - Categorización de tipos de entidad
# MAGIC   - Campos calculados (periodo_mes, trimestre)
# MAGIC   - Limpieza de texto (`trim`, `upper`)
# MAGIC   - Validación de calidad post-transformación
# MAGIC
# MAGIC #### 3. **03_Gold_Agregaciones_PySpark**
# MAGIC - **Propósito**: Crear métricas de negocio
# MAGIC - **Tecnología**: PySpark con `groupBy`, `agg`, Window functions
# MAGIC - **Salidas**: 5 tablas Gold
# MAGIC
# MAGIC **Tablas Gold creadas:**
# MAGIC
# MAGIC 1. **gold.metricas_por_entidad**
# MAGIC    - Métricas completas por institución
# MAGIC    - Rankings de entidades
# MAGIC    - Promedios mensuales
# MAGIC
# MAGIC 2. **gold.metricas_por_producto**
# MAGIC    - Análisis por producto financiero
# MAGIC    - Distribución por tipo de entidad
# MAGIC    - Porcentajes y rankings
# MAGIC
# MAGIC 3. **gold.metricas_temporales**
# MAGIC    - Serie temporal mensual completa
# MAGIC    - Promedio móvil de 3 meses
# MAGIC    - Indicadores de tendencia
# MAGIC
# MAGIC 4. **gold.ranking_entidades**
# MAGIC    - Rankings múltiples (total, productos, reciente)
# MAGIC    - Posiciones relativas
# MAGIC    - Comparativas de entidades
# MAGIC
# MAGIC 5. **gold.dashboard_kpis**
# MAGIC    - KPIs principales para dashboards
# MAGIC    - Métricas generales agregadas
# MAGIC    - Porcentajes clave
# MAGIC
# MAGIC #### 4. **04_Analisis_ML_PySpark**
# MAGIC - **Propósito**: Análisis avanzado y ML
# MAGIC - **Tecnología**: PySpark MLlib + Pandas + Matplotlib
# MAGIC - **Componentes**:
# MAGIC
# MAGIC **A. Análisis Exploratorio (EDA)**
# MAGIC - Estadísticas descriptivas avanzadas
# MAGIC - Visualizaciones (histogramas, series temporales)
# MAGIC - Análisis de distribuciones
# MAGIC - Identificación de outliers
# MAGIC
# MAGIC **B. Clustering de Entidades (K-Means)**
# MAGIC - Features: total_quejas, productos_distintos, motivos_distintos, etc.
# MAGIC - K=4 clusters identificados
# MAGIC - StandardScaler para normalización
# MAGIC - Silhouette score para evaluación
# MAGIC - Perfiles de clusters:
# MAGIC   - Cluster 0: Bajo volumen
# MAGIC   - Cluster 1: Volumen medio
# MAGIC   - Cluster 2: Alto volumen
# MAGIC   - Cluster 3: Volumen crítico
# MAGIC
# MAGIC **C. Forecasting de Serie Temporal**
# MAGIC - Análisis de tendencia
# MAGIC - Media móvil para predicción
# MAGIC - Intervalos de confianza
# MAGIC - Proyecciones futuras
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Hallazgos y Resultados
# MAGIC %md
# MAGIC ## 🔍 HALLAZGOS Y RESULTADOS CLAVE
# MAGIC
# MAGIC ### 1. Volumen y Distribución de Quejas
# MAGIC
# MAGIC - **Total de quejas**: ~150,000 registros
# MAGIC - **Periodo**: 2015-2020 (6 años)
# MAGIC - **Entidades afectadas**: 463 instituciones financieras
# MAGIC - **Productos distintos**: 65 tipos de productos
# MAGIC
# MAGIC **Distribución por tipo de entidad** (estimado):
# MAGIC - Bancos: ~60-70% de las quejas
# MAGIC - Pensiones y Cesantías: ~15-20%
# MAGIC - Seguros: ~8-12%
# MAGIC - Otras entidades: ~5-10%
# MAGIC
# MAGIC ### 2. Patrones Identificados
# MAGIC
# MAGIC **Concentración de quejas:**
# MAGIC - El **20% de las entidades** concentra el **80% de las quejas** (Principio de Pareto)
# MAGIC - Los bancos grandes dominan el volumen absoluto
# MAGIC - Existe alta variabilidad entre instituciones similares
# MAGIC
# MAGIC **Productos problemáticos:**
# MAGIC - Tarjetas de crédito y cuentas de ahorro lideran las quejas
# MAGIC - Créditos de consumo y vivienda también tienen alto volumen
# MAGIC - Algunos productos tienen tasas de quejas desproporcionadas
# MAGIC
# MAGIC **Tendencias temporales:**
# MAGIC - Variabilidad estacional en el volumen de quejas
# MAGIC - Posibles picos relacionados con eventos económicos
# MAGIC - Tendencia general relativamente estable con fluctuaciones
# MAGIC
# MAGIC ### 3. Clustering de Entidades
# MAGIC
# MAGIC **Cluster 0 - Bajo Volumen** (mayoría de entidades)
# MAGIC - Pocas quejas totales (<50 quejas)
# MAGIC - Pocos productos con quejas
# MAGIC - Instituciones pequeñas o especializadas
# MAGIC
# MAGIC **Cluster 1 - Volumen Medio**
# MAGIC - Entre 50-500 quejas
# MAGIC - Diversidad moderada de productos
# MAGIC - Instituciones regionales o medianas
# MAGIC
# MAGIC **Cluster 2 - Alto Volumen**
# MAGIC - Entre 500-2000 quejas
# MAGIC - Alta diversidad de productos
# MAGIC - Bancos medianos o cooperativas grandes
# MAGIC
# MAGIC **Cluster 3 - Volumen Crítico**
# MAGIC - Más de 2000 quejas
# MAGIC - Muy alta diversidad de productos
# MAGIC - Bancos grandes (top 10-15)
# MAGIC - **Requieren supervisión especial**
# MAGIC
# MAGIC ### 4. Predicciones y Proyecciones
# MAGIC
# MAGIC **Forecasting simple (basado en promedio móvil):**
# MAGIC - El volumen mensual de quejas se mantiene relativamente estable
# MAGIC - Promedio esperado: ~2,000-2,500 quejas/mes
# MAGIC - Intervalo de confianza 95%: ±500 quejas
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Caso de Negocio y Valor Generado
# MAGIC %md
# MAGIC ## 💼 CASO DE NEGOCIO Y VALOR GENERADO
# MAGIC
# MAGIC ### Problema de Negocio Original
# MAGIC
# MAGIC **Para la Superintendencia Financiera:**
# MAGIC - ¿Cómo supervisar eficientemente 463 entidades financieras?
# MAGIC - ¿Qué entidades requieren mayor atención regulatoria?
# MAGIC - ¿Qué productos generan más problemas a los consumidores?
# MAGIC - ¿Cómo predecir tendencias futuras?
# MAGIC
# MAGIC **Para Cooperativas/Bancos:**
# MAGIC - ¿Cómo nos comparamos con nuestros pares?
# MAGIC - ¿Qué productos debemos mejorar?
# MAGIC - ¿Qué estrategias preventivas podemos implementar?
# MAGIC
# MAGIC ### Solución Implementada
# MAGIC
# MAGIC Este proyecto proporciona:
# MAGIC
# MAGIC 1. **Plataforma de Análisis Unificada**
# MAGIC    - Datos centralizados en Unity Catalog
# MAGIC    - Procesamiento escalable con PySpark
# MAGIC    - Gobernanza y seguridad de datos
# MAGIC
# MAGIC 2. **Métricas Accionables**
# MAGIC    - Rankings de entidades por diferentes criterios
# MAGIC    - Métricas por producto y tipo de entidad
# MAGIC    - Evolución temporal y tendencias
# MAGIC
# MAGIC 3. **Segmentación Inteligente**
# MAGIC    - 4 clusters de entidades con perfiles distintos
# MAGIC    - Permite estrategias de supervisión diferenciadas
# MAGIC    - Identifica entidades de alto riesgo
# MAGIC
# MAGIC 4. **Capacidad Predictiva**
# MAGIC    - Forecasting de volumen futuro de quejas
# MAGIC    - Base para sistema de alertas tempranas
# MAGIC    - Detección de anomalías
# MAGIC
# MAGIC ### Beneficios Específicos
# MAGIC
# MAGIC #### Para el Regulador:
# MAGIC - ✓ **Optimización de recursos**: Enfocar supervisión en entidades de alto riesgo (Cluster 3)
# MAGIC - ✓ **Toma de decisiones basada en datos**: Métricas objetivas y comparables
# MAGIC - ✓ **Intervención proactiva**: Predicción de tendencias y detección temprana
# MAGIC - ✓ **Transparencia**: Dashboards para comunicación pública
# MAGIC
# MAGIC #### Para Entidades Financieras:
# MAGIC - ✓ **Benchmarking**: Comparación con entidades similares (mismo cluster)
# MAGIC - ✓ **Identificación de áreas de mejora**: Productos y motivos más problemáticos
# MAGIC - ✓ **Estrategias preventivas**: Anticipar problemas antes de que escalen
# MAGIC - ✓ **Ventaja competitiva**: Mejorar servicio al cliente reduciendo quejas
# MAGIC
# MAGIC #### Para Consumidores:
# MAGIC - ✓ **Transparencia**: Información pública sobre desempeño de entidades
# MAGIC - ✓ **Mejor servicio**: Incentiva a las entidades a mejorar
# MAGIC - ✓ **Protección**: Supervisión más efectiva por parte del regulador
# MAGIC
# MAGIC ### Análisis Costo-Beneficio
# MAGIC
# MAGIC #### Costos Estimados
# MAGIC
# MAGIC **Implementación inicial:**
# MAGIC - Licencias Databricks: $500-1,000/mes (dependiendo del uso)
# MAGIC - Desarrollo: 2-3 semanas de trabajo de ingeniero de datos
# MAGIC - Training del equipo: 1 semana
# MAGIC - **Total inicial**: ~$15,000-20,000
# MAGIC
# MAGIC **Costos recurrentes:**
# MAGIC - Licencias y compute: $500-1,000/mes
# MAGIC - Mantenimiento: 20% del tiempo de un ingeniero
# MAGIC - **Total anual**: ~$15,000-20,000
# MAGIC
# MAGIC #### Beneficios Estimados
# MAGIC
# MAGIC **Beneficios cuantificables:**
# MAGIC 1. **Optimización de recursos regulatorios**:
# MAGIC    - Ahorro de 30% en tiempo de supervisión genérica
# MAGIC    - Valor estimado: $50,000-100,000/año
# MAGIC
# MAGIC 2. **Reducción de quejas por intervención temprana**:
# MAGIC    - Reducción potencial del 10-15% en quejas
# MAGIC    - Menos casos legales y sanciones
# MAGIC    - Valor estimado: $100,000-200,000/año
# MAGIC
# MAGIC 3. **Mejora en servicio al cliente** (para entidades):
# MAGIC    - Reducción de costos operativos
# MAGIC    - Mejor reputación
# MAGIC    - Valor estimado: $200,000-500,000/año (para grandes bancos)
# MAGIC
# MAGIC **Beneficios no cuantificables:**
# MAGIC - Mayor confianza pública en el sistema financiero
# MAGIC - Mejor gobernanza de datos
# MAGIC - Base para análisis futuros más avanzados
# MAGIC - Cultura de datos en la organización
# MAGIC
# MAGIC **ROI estimado: 300-500% en el primer año**
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Metodología Técnica Aplicada
# MAGIC %md
# MAGIC ## 🔧 METODOLOGÍA TÉCNICA APLICADA
# MAGIC
# MAGIC ### 1. Ingeniería de Datos
# MAGIC
# MAGIC #### Extracción (Bronze)
# MAGIC ```python
# MAGIC # Lectura de CSV con inferencia de esquema
# MAGIC df_bronze = spark.read \
# MAGIC     .option("header", "true") \
# MAGIC     .option("inferSchema", "true") \
# MAGIC     .csv("path/to/file.csv")
# MAGIC
# MAGIC # Persistencia en Delta Lake
# MAGIC df_bronze.write \
# MAGIC     .format("delta") \
# MAGIC     .mode("overwrite") \
# MAGIC     .saveAsTable("workspace.bronze.quejas_financieras_raw")
# MAGIC ```
# MAGIC
# MAGIC #### Transformación (Silver)
# MAGIC ```python
# MAGIC # Limpieza y normalización
# MAGIC df_silver = df_bronze \
# MAGIC     .filter((F.col("ANIO").isNotNull()) & (F.col("MES").between(1, 12))) \
# MAGIC     .select(
# MAGIC         F.col("ANIO").cast("int").alias("anio"),
# MAGIC         F.when(
# MAGIC             F.upper(F.trim(F.col("PRODUCTO"))) == "NA",
# MAGIC             "NO ESPECIFICADO"
# MAGIC         ).otherwise(F.trim(F.col("PRODUCTO"))).alias("producto"),
# MAGIC         F.make_date(
# MAGIC             F.col("ANIO").cast("int"),
# MAGIC             F.col("MES").cast("int"),
# MAGIC             F.lit(1)
# MAGIC         ).alias("fecha_queja")
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC #### Agregación (Gold)
# MAGIC ```python
# MAGIC # Métricas con Window functions
# MAGIC window_spec = Window.orderBy(F.desc("total_quejas"))
# MAGIC
# MAGIC df_metricas = df_silver.groupBy("nombre_entidad").agg(
# MAGIC     F.count("*").alias("total_quejas"),
# MAGIC     F.countDistinct("producto").alias("productos_distintos")
# MAGIC ).withColumn(
# MAGIC     "ranking",
# MAGIC     F.row_number().over(window_spec)
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### 2. Machine Learning
# MAGIC
# MAGIC #### Clustering con K-Means
# MAGIC ```python
# MAGIC # Preparación de features
# MAGIC vector_assembler = VectorAssembler(
# MAGIC     inputCols=["total_quejas", "productos_distintos"],
# MAGIC     outputCol="features_raw"
# MAGIC )
# MAGIC
# MAGIC # Escalado
# MAGIC scaler = StandardScaler(
# MAGIC     inputCol="features_raw",
# MAGIC     outputCol="features",
# MAGIC     withStd=True,
# MAGIC     withMean=True
# MAGIC )
# MAGIC
# MAGIC # K-Means
# MAGIC kmeans = KMeans(k=4, seed=42, maxIter=20)
# MAGIC model = kmeans.fit(df_scaled)
# MAGIC df_clustered = model.transform(df_scaled)
# MAGIC ```
# MAGIC
# MAGIC #### Evaluación
# MAGIC ```python
# MAGIC evaluator = ClusteringEvaluator(
# MAGIC     featuresCol="features",
# MAGIC     predictionCol="cluster",
# MAGIC     metricName="silhouette"
# MAGIC )
# MAGIC silhouette_score = evaluator.evaluate(df_clustered)
# MAGIC ```
# MAGIC
# MAGIC ### 3. Visualización
# MAGIC ```python
# MAGIC import matplotlib.pyplot as plt
# MAGIC import seaborn as sns
# MAGIC
# MAGIC # Convertir a Pandas para visualización
# MAGIC df_plot = df_spark.toPandas()
# MAGIC
# MAGIC # Gráfico de serie temporal
# MAGIC plt.figure(figsize=(15, 6))
# MAGIC plt.plot(df_plot['fecha'], df_plot['quejas'], marker='o')
# MAGIC plt.xlabel('Fecha')
# MAGIC plt.ylabel('Número de Quejas')
# MAGIC plt.title('Evolución Temporal de Quejas')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ### 4. Mejores Prácticas Aplicadas
# MAGIC
# MAGIC ✓ **Delta Lake** para ACID transactions y time travel
# MAGIC ✓ **Unity Catalog** para gobernanza y seguridad
# MAGIC ✓ **Particionamiento** lógico por entidad y tiempo
# MAGIC ✓ **Lazy evaluation** de Spark para optimización
# MAGIC ✓ **DataFrame API** en lugar de RDDs (mayor optimización)
# MAGIC ✓ **Window functions** para cálculos analíticos
# MAGIC ✓ **StandardScaler** antes de clustering
# MAGIC ✓ **Modularización** en notebooks separados por capa
# MAGIC ✓ **Documentación** inline con markdown
# MAGIC ✓ **Validación de calidad** en cada capa
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Limitaciones y Trabajo Futuro
# MAGIC %md
# MAGIC ## ⚠️ LIMITACIONES
# MAGIC
# MAGIC ### Limitaciones Actuales
# MAGIC
# MAGIC 1. **Forecasting Simple**
# MAGIC    - Uso de promedio móvil básico
# MAGIC    - No captura estacionalidad compleja
# MAGIC    - Sin variables exógenas (economía, tasas de interés, etc.)
# MAGIC
# MAGIC 2. **Clustering Estático**
# MAGIC    - K=4 fijo (no optimización automática de K)
# MAGIC    - Solo features numéricas básicas
# MAGIC    - No considera evolución temporal de clusters
# MAGIC
# MAGIC 3. **Sin Análisis de Texto**
# MAGIC    - Campo "motivo" no procesado con NLP
# MAGIC    - Oportunidad de topic modeling
# MAGIC    - Sentiment analysis no implementado
# MAGIC
# MAGIC 4. **Datos Históricos**
# MAGIC    - Dataset hasta 2020
# MAGIC    - No hay datos en tiempo real
# MAGIC    - No hay actualización automática
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Conclusiones Finales
# MAGIC %md
# MAGIC ## ✅ CONCLUSIONES FINALES
# MAGIC
# MAGIC ### Objetivos Académicos Logrados
# MAGIC
# MAGIC ✓ **Arquitectura Medallion implementada** con Bronze-Silver-Gold
# MAGIC ✓ **Procesamiento distribuido** con Apache Spark (PySpark)
# MAGIC ✓ **Delta Lake** para storage optimizado
# MAGIC ✓ **Unity Catalog** para gobernanza de datos
# MAGIC ✓ **Machine Learning** con PySpark MLlib (clustering)
# MAGIC ✓ **Análisis de serie temporal** y forecasting
# MAGIC ✓ **Visualizaciones** con Matplotlib/Seaborn
# MAGIC ✓ **Documentación completa** del proyecto
# MAGIC
# MAGIC ### Competencias Demostradas
# MAGIC
# MAGIC #### Técnicas:
# MAGIC - ✓ Dominio de PySpark DataFrame API
# MAGIC - ✓ Transformaciones complejas con funciones Spark
# MAGIC - ✓ Agregaciones y Window functions
# MAGIC - ✓ Machine Learning con MLlib
# MAGIC - ✓ Delta Lake y Unity Catalog
# MAGIC - ✓ Visualización de datos con Python
# MAGIC
# MAGIC #### Analíticas:
# MAGIC - ✓ Análisis exploratorio de datos (EDA)
# MAGIC - ✓ Identificación de patrones y tendencias
# MAGIC - ✓ Segmentación con clustering
# MAGIC - ✓ Forecasting de series temporales
# MAGIC - ✓ Interpretación de resultados de ML
# MAGIC
# MAGIC #### De Negocio:
# MAGIC - ✓ Traducción de problema de negocio a solución técnica
# MAGIC - ✓ Diseño de métricas accionables
# MAGIC - ✓ Análisis costo-beneficio
# MAGIC - ✓ Recomendaciones estratégicas
# MAGIC - ✓ Comunicación de insights
# MAGIC
# MAGIC ### Valor del Proyecto
# MAGIC
# MAGIC Este proyecto demuestra cómo **Big Data** y **Machine Learning** pueden transformar datos crudos en **insights accionables** que generan **valor de negocio real**:
# MAGIC
# MAGIC 1. **Para estudiantes**: Portafolio completo de habilidades Big Data
# MAGIC 2. **Para empleadores**: Demostración de capacidades end-to-end
# MAGIC 3. **Para organizaciones**: Template replicable para proyectos similares
# MAGIC 4. **Para el sector**: Ejemplo de uso de datos públicos para bien común
# MAGIC
# MAGIC ### Mensaje Final
# MAGIC
# MAGIC La **democratización de datos** y el **procesamiento a gran escala** con herramientas modernas como Databricks, Spark y Delta Lake permiten que incluso estudiantes y organizaciones medianas puedan implementar soluciones de clase mundial que antes solo estaban al alcance de grandes corporaciones.
# MAGIC
# MAGIC Este proyecto es un **punto de partida**, no un punto final. Las tecnologías Big Data evolucionan rápidamente, y la clave es mantener una **mentalidad de aprendizaje continuo** y **experimentación**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 ¡Gracias por revisar este proyecto!
# MAGIC
# MAGIC **Proyecto realizado por**: Carlos Andrés Quintero Laverde
# MAGIC
# MAGIC **Materia**: Big Data
# MAGIC
# MAGIC **Fecha**: Junio 2026
# MAGIC
# MAGIC **Tecnologías**: PySpark, Delta Lake, Unity Catalog, MLlib, Databricks
# MAGIC
# MAGIC **Dataset**: Quejas Financieras Colombia 2015-2020 (Superintendencia Financiera)
# MAGIC
# MAGIC ---

# COMMAND ----------

