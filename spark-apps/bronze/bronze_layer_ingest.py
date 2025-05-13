from pyspark.sql import SparkSession
from pyspark.sql.functions import current_date, current_timestamp


def create_spark_session():
    return (
        SparkSession.builder
        .appName("BronzeLayerIngestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def ingest_table(spark, table_name: str, raw_base: str, bronze_base: str):
    """
    Reads a raw CSV table and writes it to Delta bronze layer,
    adding ingest metadata and partitioning by ingest_date.
    """
    raw_path = f"{raw_base}/{table_name}.csv"
    bronze_path = f"{bronze_base}/{table_name}"

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(raw_path)
    )
    # Add metadata
    df = (
        df.withColumn("ingest_date", current_date())
          .withColumn("ingest_timestamp", current_timestamp())
    )

    # Write to Delta
    (
        df.write
          .format("delta")
          .mode("overwrite")
          .option("overwriteSchema", True)
          .partitionBy("ingest_date")
          .save(bronze_path)
    )


if __name__ == "__main__":
    spark = create_spark_session()

    # S3/MinIO paths
    RAW_BASE = "s3a://lakehouse/raw"
    BRONZE_BASE = "s3a://lakehouse/bronze"

    tables = [
        "application_train",
        "application_test",
        "bureau",
        "bureau_balance",
        "previous_application",
        "POS_CASH_balance",
        "credit_card_balance",
        "installments_payments",
    ]

    for tbl in tables:
        ingest_table(spark, tbl, RAW_BASE, BRONZE_BASE)

    spark.stop()
