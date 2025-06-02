from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    count as spark_count,
    sum   as spark_sum,
    avg   as spark_avg,
    round as spark_round,
    col, date_format, expr
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


def extract_contract_summary(spark, gold_base: str):
    """
    Build and persist contract summary analytics.
    """
    dim_contract = spark.read.format("delta").load(f"{gold_base}/dim_contract")
    dim_date     = spark.read.format("delta").load(f"{gold_base}/dim_date")

    # Join to enrich with calendar fields
    df = dim_contract.alias("c") \
        .join(
            dim_date.alias("d"),
            col("c.application_date_id") == col("d.date_id"),
            "left"
        )

    # Calculate summary metrics by month and contract type
    contract_summary = (
        df
        .groupBy(
            col("d.year").alias("year"),
            col("d.month").alias("month"),
            col("c.name_contract_type")
        )
        .agg(
            spark_count("c.sk_id_curr").alias("cnt_contracts"),
            spark_round(spark_sum("c.amt_credit"), 2).alias("sum_credit"),
            spark_round(spark_avg("c.credit_income_ratio"), 4)
            .alias("avg_credit_income_ratio"),
            spark_round(
                spark_avg(col("c.amt_credit") / col("c.amt_annuity")),
                2
            ).alias("avg_term_months")
        )
        .withColumn(
            "month_year",
            date_format(
                expr("make_date(year, month, 1)"),
                "yyyy-MM"
            )
        )
        .select(
            "month_year",
            "name_contract_type",
            "cnt_contracts",
            "sum_credit",
            "avg_credit_income_ratio",
            "avg_term_months"
        )
    )

    # Write out to Gold as Delta
    out_path = f"{gold_base}/contract_summary"
    contract_summary.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(out_path)
    print(f"Written contract_summary to {out_path}")

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
    # copy_dim_conformed(spark, SILVER_BASE, GOLD_BASE, dims_to_copy)

    # KHAI THAC DU LIEU
    extract_contract_summary(spark, GOLD_BASE)


    spark.stop()
