from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    expr,
    year,
    month,
    dayofmonth,
    dayofweek,
    floor,
    lit,
    to_date,
    date_add,
    when,
    date_format,
    add_months,
    round as spark_round
)

def create_spark_session():
    return (
        SparkSession.builder
        .appName("GoldLayerIngestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

def copy_dim_conformed(spark, silver_base: str, gold_base: str, dims: list):
    """
    Copy dimension tables from Silver to Gold, preserving schema.
    """
    for dim in dims:
        src = f"{silver_base}/{dim}"
        tgt = f"{gold_base}/{dim}"
        df = spark.read.format("delta").load(src)
        df.write \
          .format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", True) \
          .save(tgt)
        print(f"Copied dim table '{dim}' from {src} to {tgt}")

if __name__ == "__main__":
    spark = create_spark_session()

    SILVER_BASE = "s3a://lakehouse/silver"
    GOLD_BASE   = "s3a://lakehouse/gold"

    # List of conformed dimension tables to copy
    dims_to_copy = [
        "dim_date",
        "dim_client",
        "dim_contract",
        "dim_employment",
        "dim_ext_source",
        "dim_document_flags",
        "dim_cb_requests"
    ]
    copy_dim_conformed(spark, SILVER_BASE, GOLD_BASE, dims_to_copy)

    spark.stop()
