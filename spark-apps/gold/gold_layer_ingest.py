from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    count as spark_count,
    sum   as spark_sum,
    min  as spark_min,
    avg   as spark_avg,
    round as spark_round,
    year, substring, col, date_format, expr, lit
)

def create_spark_session():
    return (
        SparkSession.builder
        .appName("GoldLayerIngestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


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


def build_gold_origination(spark, silver_base, gold_base):
    # Load silver tables
    fact_app       = spark.read.format("delta").load(f"{silver_base}/fact_application")
    dim_date       = spark.read.format("delta").load(f"{silver_base}/dim_date")
    dim_contract   = spark.read.format("delta").load(f"{silver_base}/dim_contract")
    dim_client     = spark.read.format("delta").load(f"{silver_base}/dim_client")
    dim_employment = spark.read.format("delta").load(f"{silver_base}/dim_employment")
    dim_ext        = spark.read.format("delta").load(f"{silver_base}/dim_ext_source")
    dim_cb         = spark.read.format("delta").load(f"{silver_base}/dim_cb_requests")

    # 1. Số lượng hồ sơ theo tháng-năm và loại hợp đồng
    df_count = (
        fact_app
        .join(dim_date, fact_app.application_date_id == dim_date.date_id)
        .join(dim_contract, ["sk_id_curr", "application_date_id"])
        .withColumn("month_year", date_format(col("application_date"), "yyyy-MM"))
        .groupBy(col("month_year"), col("NAME_CONTRACT_TYPE").alias("contract_type"))
        .agg(spark_count("*").alias("num_applications"))
    )
    df_count.write.format("delta").mode("overwrite").save(f"{gold_base}/origination_count")

    # 2. Tổng giá trị giải ngân theo region và nghề nghiệp
    df_sum = (
        fact_app
        .join(dim_contract, ["sk_id_curr", "application_date_id"])
        .join(dim_client, "sk_id_curr")
        .join(dim_employment, ["sk_id_curr", "application_date_id"])
        .groupBy(
            col("REGION_POPULATION_RELATIVE").alias("region"),
            col("OCCUPATION_TYPE").alias("occupation")
        )
        .agg(spark_sum(col("AMT_CREDIT")).alias("total_disbursed"))
    )
    df_sum.write.format("delta").mode("overwrite").save(f"{gold_base}/origination_sum")

    # 3. Giá trị trung bình hồ sơ theo loại thu nhập
    df_avg = (
        fact_app
        .join(dim_contract, ["sk_id_curr", "application_date_id"])
        .join(dim_client, "sk_id_curr")
        .groupBy(col("NAME_INCOME_TYPE").alias("income_type"))
        .agg(spark_avg(col("AMT_CREDIT")).alias("avg_loan_amount"))
    )
    df_avg.write.format("delta").mode("overwrite").save(f"{gold_base}/origination_avg")

    # 4. Tỷ lệ phê duyệt theo external score và số lần yêu cầu CB trong năm
    df_appr = (
        fact_app
        .join(dim_ext, ["sk_id_curr", "application_date_id"])
        .join(dim_cb, ["sk_id_curr", "application_date_id"])
        .groupBy(
            col("ext_source_mean"),
            col("AMT_REQ_CREDIT_BUREAU_YEAR").alias("cb_requests_year")
        )
        .agg(
            (spark_sum(expr("CASE WHEN target = 0 THEN 1 ELSE 0 END")) / spark_count("*")).alias("approval_rate")
        )
    )
    df_appr.write.format("delta").mode("overwrite").save(f"{gold_base}/origination_approval_rate")



def build_gold_portfolio_performance(spark, silver_base, gold_base):
    # Load silver tables
    fact_cc      = spark.read.format("delta").load(f"{silver_base}/fact_credit_card_balance")
    fact_pos     = spark.read.format("delta").load(f"{silver_base}/fact_pos_cash_balance")
    fact_prev    = spark.read.format("delta").load(f"{silver_base}/fact_previous_application")
    dim_client   = spark.read.format("delta").load(f"{silver_base}/dim_client")
    dim_contract = spark.read.format("delta").load(f"{silver_base}/dim_contract")

    # Prepare contract lookup with unique column name
    contract_df = dim_contract.select(
        col("sk_id_curr"),
        col("application_date_id"),
        col("NAME_CONTRACT_TYPE").alias("contract_type")
    )

    # 1. Tổng dư nợ đang hoạt động (active balance) theo region
    df_cc = fact_cc.groupBy("sk_id_curr").agg(spark_sum(col("amt_balance")).alias("cc_balance"))
    df_pos = fact_pos.groupBy("sk_id_curr").agg(spark_sum(col("cnt_instalment")).alias("pos_balance"))
    df_total = (
        df_cc
        .join(df_pos, "sk_id_curr", "outer")
        .na.fill(0)
        .withColumn("total_balance", col("cc_balance") + col("pos_balance"))
        .join(dim_client, "sk_id_curr")
        .groupBy(col("REGION_POPULATION_RELATIVE").alias("region"))
        .agg(spark_sum("total_balance").alias("total_active_balance"))
    )
    df_total.write.format("delta").mode("overwrite").save(f"{gold_base}/portfolio_total_active_balance")

    # 2. Tỷ lệ sử dụng hạn mức theo loại thu nhập (income type)
    df_util = (
        fact_cc
        .join(dim_client, "sk_id_curr")
        .groupBy(col("NAME_INCOME_TYPE").alias("client_segment"))
        .agg(
            (spark_sum(col("amt_balance")) / spark_sum(col("amt_credit_limit_actual"))).alias("utilization_ratio")
        )
    )
    df_util.write.format("delta").mode("overwrite").save(f"{gold_base}/portfolio_utilization_ratio")

    # 3. Số dư nợ quá hạn (DPD > 0) theo tháng offset
    df_overdue = (
        fact_cc
        .filter(col("sk_dpd") > 0)
        .groupBy(col("months_balance").alias("month_offset"))
        .agg(spark_sum(col("amt_balance")).alias("overdue_balance"))
    )
    df_overdue.write.format("delta").mode("overwrite").save(f"{gold_base}/portfolio_overdue_balance")

    # 4. Tỷ lệ tài sản quá hạn theo loại hợp đồng (credit type)
    df_rate = (
        fact_cc
        .join(fact_prev, ["sk_id_curr", "sk_id_prev"])  # map to previous application
        .join(contract_df, ["sk_id_curr", "application_date_id"])  # get contract_type uniquely
        .groupBy(col("contract_type"))
        .agg(
            (spark_sum(expr("CASE WHEN sk_dpd > 0 THEN 1 ELSE 0 END")) / spark_count("*")).alias("overdue_ratio")
        )
    )
    df_rate.write.format("delta").mode("overwrite").save(f"{gold_base}/portfolio_overdue_ratio")

# -------- Delinquency & Default Metrics --------
def build_gold_delinquency_default(spark, silver_base, gold_base):
    # Load silver tables
    fact_app      = spark.read.format("delta").load(f"{silver_base}/fact_application")
    fact_inst     = spark.read.format("delta").load(f"{silver_base}/fact_installments_payments")
    fact_prev     = spark.read.format("delta").load(f"{silver_base}/fact_previous_application")
    dim_contract  = spark.read.format("delta").load(f"{silver_base}/dim_contract")
    dim_client    = spark.read.format("delta").load(f"{silver_base}/dim_client")
    dim_employment= spark.read.format("delta").load(f"{silver_base}/dim_employment")
    dim_ext       = spark.read.format("delta").load(f"{silver_base}/dim_ext_source")

    # Map previous application date
    prev_df = fact_prev.select(
        col("sk_id_curr"), col("sk_id_prev"), col("application_date_id")
    )

    # 1. Default rate by contract type
    df_default = (
        fact_app
        .join(dim_contract, ["sk_id_curr","application_date_id"])
        .groupBy(col("NAME_CONTRACT_TYPE").alias("contract_type"))
        .agg((spark_sum(expr("CASE WHEN target = 1 THEN 1 ELSE 0 END")) / spark_count("*")).alias("default_rate"))
    )
    df_default.write.format("delta").mode("overwrite").save(f"{gold_base}/delinquency_default_default_rate")

    # 2. Delinquency rates 30/60/90+ by family status
    inst_client = fact_inst.join(dim_client, "sk_id_curr")
    total_per_status = inst_client.groupBy(col("NAME_FAMILY_STATUS").alias("family_status")).agg(spark_count("*").alias("total_count"))
    for thr in [30, 60, 90]:
        overdue = (
            inst_client.filter(col("days_instalment") > thr)
                .groupBy(col("NAME_FAMILY_STATUS").alias("family_status"))
                .agg(spark_count("*").alias("overdue_count"))
        )
        df_rate = (
            overdue.join(total_per_status, "family_status")
                .select(
                    col("family_status"),
                    (col("overdue_count")/col("total_count")).alias(f"delinq_{thr}")
                )
        )
        df_rate.write.format("delta").mode("overwrite").save(f"{gold_base}/delinquency_default_rate_{thr}")

    # 3. Average days past due by occupation
    df_avg_days = (
        fact_inst.filter(col("days_entry_payment") > 0)
            .join(prev_df, ["sk_id_curr","sk_id_prev"])
            .join(dim_employment, ["sk_id_curr","application_date_id"])
            .groupBy(col("OCCUPATION_TYPE").alias("occupation"))
            .agg(spark_avg(col("days_entry_payment")).alias("avg_days_past_due"))
    )
    df_avg_days.write.format("delta").mode("overwrite").save(f"{gold_base}/delinquency_avg_days_past_due")

    # 4. Time to first default by ext source
    df_first = (
        fact_inst.filter(col("days_instalment") > 0)
            .join(prev_df, ["sk_id_curr","sk_id_prev"])
            .join(dim_ext, ["sk_id_curr","application_date_id"])
            .groupBy(col("ext_source_mean"))
            .agg(spark_min(col("days_instalment")).alias("time_to_first_default"))
    )
    df_first.write.format("delta").mode("overwrite").save(f"{gold_base}/delinquency_time_to_first_default")

# -------- Credit Bureau Metrics --------
def build_gold_credit_bureau(spark, silver_base, gold_base):
    fact_bur   = spark.read.format("delta").load(f"{silver_base}/fact_bureau")
    dim_client = spark.read.format("delta").load(f"{silver_base}/dim_client")
    dim_cb     = spark.read.format("delta").load(f"{silver_base}/dim_cb_requests")

    # 1. Số khoản vay cũ theo region
    df_count_old = (
        fact_bur
        .join(dim_client, "sk_id_curr")
        .groupBy(col("REGION_POPULATION_RELATIVE").alias("region"))
        .agg(spark_count("*").alias("old_loans_count"))
    )
    df_count_old.write.format("delta").mode("overwrite").save(f"{gold_base}/cb_old_loans_count")

    # 2. Tổng dư nợ cũ theo năm
    df_sum_old = (
        fact_bur
        .withColumn("year", year(col("credit_date")))  # extract year from date
        .groupBy("year")
        .agg(spark_sum(col("amt_credit_sum")).alias("old_credit_sum"))
    )
    df_sum_old.write.format("delta").mode("overwrite").save(f"{gold_base}/cb_old_credit_sum_by_year")

    # 3. Số khoản vay gia hạn theo số lần request CB trong năm
    df_prolong = (
        fact_bur
        .join(dim_cb, "sk_id_curr")
        .groupBy(col("AMT_REQ_CREDIT_BUREAU_YEAR").alias("cb_requests_year"))
        .agg(spark_sum(col("cnt_credit_prolong")).alias("total_credit_prolong"))
    )
    df_prolong.write.format("delta").mode("overwrite").save(f"{gold_base}/cb_credit_prolong_by_requests_year")

    # 4. Tổng số lần request CB theo tháng (YYYYMM)
    df_requests = (
        dim_cb
        .withColumn("month_year", substring(col("application_date_id").cast("string"), 1, 6))  # YYYYMM
        .groupBy("month_year")
        .agg(
            (
                spark_sum(col("AMT_REQ_CREDIT_BUREAU_HOUR"))
                + spark_sum(col("AMT_REQ_CREDIT_BUREAU_DAY"))
                + spark_sum(col("AMT_REQ_CREDIT_BUREAU_WEEK"))
                + spark_sum(col("AMT_REQ_CREDIT_BUREAU_MON"))
                + spark_sum(col("AMT_REQ_CREDIT_BUREAU_QRT"))
                + spark_sum(col("AMT_REQ_CREDIT_BUREAU_YEAR"))
            ).alias("total_cb_requests")
        )
    )
    df_requests.write.format("delta").mode("overwrite").save(f"{gold_base}/cb_total_requests_by_month")



if __name__ == "__main__":
    spark = create_spark_session()

    SILVER_BASE = "s3a://lakehouse/silver"
    GOLD_BASE   = "s3a://lakehouse/gold"

    # KHAI THAC DU LIEU
    # extract_contract_summary(spark, GOLD_BASE)
    # build_gold_origination(spark, SILVER_BASE, GOLD_BASE)
    # build_gold_portfolio_performance(spark, SILVER_BASE, GOLD_BASE)
    # build_gold_delinquency_default(spark, SILVER_BASE, GOLD_BASE)
    # build_gold_credit_bureau(spark, SILVER_BASE, GOLD_BASE)
    spark.stop()
