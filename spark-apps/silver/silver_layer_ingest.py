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
        .appName("SilverLayerIngestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def build_dim_date(spark, bronze_base, silver_base):
    df_app = spark.read.format("delta").load(f"{bronze_base}/application_train")

    df_dates = (
        df_app
        # Dùng ingest_date làm application_date
        .select(col("ingest_date").alias("application_date"))
        .distinct()
        .withColumn("date_id", date_format("application_date", "yyyyMMdd").cast("int"))
        .withColumn("year", year("application_date"))
        .withColumn("month", month("application_date"))
        .withColumn("day", dayofmonth("application_date"))
        .withColumn("weekday", dayofweek("application_date"))
    )

    df_dates.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/dim_date")


def build_dim_client(spark, bronze_base, silver_base):
    df = spark.read.format("delta").load(f"{bronze_base}/application_train")

    df_dim = (
        df
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            # Demographics
            col("CODE_GENDER"),
            col("CNT_CHILDREN").cast("int").alias("cnt_children"),
            col("FLAG_OWN_CAR").cast("int"),
            col("FLAG_OWN_REALTY").cast("int"),
            # Income & education
            col("NAME_INCOME_TYPE"),
            col("NAME_EDUCATION_TYPE"),
            # Family & housing
            col("NAME_FAMILY_STATUS"),
            col("CNT_FAM_MEMBERS").cast("int"),
            col("NAME_HOUSING_TYPE"),
            # Region
            col("REGION_POPULATION_RELATIVE").cast("double"),
            col("REGION_RATING_CLIENT").cast("int"),
            col("REGION_RATING_CLIENT_W_CITY").cast("int")
        )
        .distinct()
    )

    df_dim.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/dim_client")


def build_dim_contract(spark, bronze_base, silver_base):
    """
    Xây dựng dim_contract chỉ với các thuộc tính đúng của mỗi hợp đồng application.
    Khóa: (sk_id_curr, application_date_id)
    """
    df = spark.read.format("delta").load(f"{bronze_base}/application_train")

    df_dim = (
        df
        # Application date ID từ ingest_date
        .withColumn(
            "application_date_id",
            date_format(col("ingest_date"), "yyyyMMdd").cast("int")
        )
        # Tính tỉ lệ vay/trong thu nhập, trả góp/vay
        .withColumn(
            "credit_income_ratio",
            spark_round(col("AMT_CREDIT") / col("AMT_INCOME_TOTAL"), 4)
        )
        .withColumn(
            "annuity_credit_ratio",
            spark_round(col("AMT_ANNUITY") / col("AMT_CREDIT"), 4)
        )
        # Chọn đúng những trường “thuộc về hợp đồng”
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            "application_date_id",

            # Loại hợp đồng, ai đi cùng
            "NAME_CONTRACT_TYPE",
            "NAME_TYPE_SUITE",

            # Thông tin số tiền
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",

            # Derived features thuần hợp đồng
            "credit_income_ratio",
            "annuity_credit_ratio"
        )
        .distinct()
    )

    df_dim.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/dim_contract")


def build_dim_employment(spark, bronze_base, silver_base):
    """
    Xây dựng bảng dim_employment với:
      - application_date_id: ingest_date theo format YYYYMMDD
      - employment_years: số năm đã làm việc tính đến ingest_date
      - occupation_type, organization_type
    Khóa: sk_id_curr, application_date_id
    """
    df = spark.read.format("delta").load(f"{bronze_base}/application_train")

    df_dim = (
        df
        # application_date_id từ ingest_date
        .withColumn(
            "application_date_id",
            date_format(col("ingest_date"), "yyyyMMdd").cast("int")
        )
        # employment_years = floor(-DAYS_EMPLOYED/365)
        .withColumn(
            "employment_years",
            floor(-col("DAYS_EMPLOYED") / lit(365)).cast("int")
        )
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            "application_date_id",
            "employment_years",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE"
        )
        .distinct()
    )

    df_dim.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/dim_employment")


def build_dim_ext_source(spark, bronze_base, silver_base):
    df = spark.read.format("delta").load(f"{bronze_base}/application_train")

    df_dim = (
        df
        .withColumn(
            "application_date_id",
            date_format(col("ingest_date"), "yyyyMMdd").cast("int")
        )
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            "application_date_id",
            col("EXT_SOURCE_1").cast("double"),
            col("EXT_SOURCE_2").cast("double"),
            col("EXT_SOURCE_3").cast("double"),
            # Derived features
            ((col("EXT_SOURCE_1") + col("EXT_SOURCE_2") + col("EXT_SOURCE_3")) / lit(3)).alias("ext_source_mean"),
            expr("greatest(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)").alias("ext_source_max")
        )
        .distinct()
    )

    df_dim.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/dim_ext_source")


def build_dim_document_flags(spark, bronze_base, silver_base):
    df = spark.read.format("delta").load(f"{bronze_base}/application_train")

    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
    sum_expr = " + ".join(doc_cols)

    df_dim = (
        df
        .withColumn(
            "application_date_id",
            date_format(col("ingest_date"), "yyyyMMdd").cast("int")
        )
        .withColumn("num_docs_provided", expr(sum_expr).cast("int"))
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            "application_date_id",
            *[col(c).cast("int") for c in doc_cols],
            "num_docs_provided"
        )
        .distinct()
    )

    df_dim.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/dim_document_flags")


def build_dim_cb_requests(spark, bronze_base, silver_base):
    df = spark.read.format("delta").load(f"{bronze_base}/application_train")

    req_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR"
    ]

    df_dim = (
        df
        .withColumn(
            "application_date_id",
            date_format(col("ingest_date"), "yyyyMMdd").cast("int")
        )
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            "application_date_id",
            *[col(c).cast("int") for c in req_cols]
        )
        .distinct()
    )

    df_dim.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/dim_cb_requests")


def build_fact_application(spark, bronze_base, silver_base):
    """
    Xây dựng fact_application chỉ với:
      - sk_id_curr       : FK → dim_client
      - application_date_id : FK → dim_date / dim_contract
      - target           : measure duy nhất
    """
    df = spark.read.format("delta").load(f"{bronze_base}/application_train")

    df_fact = (
        df
        .withColumn("application_date_id",
                    date_format(col("ingest_date"), "yyyyMMdd").cast("int")
                    )
        .withColumn("application_year",
                    date_format(col("ingest_date"), "yyyy").cast("int")
                    )
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            "application_date_id",
            col("TARGET").alias("target")
        )
    )

    df_fact.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/fact_application")

def build_fact_bureau(spark, bronze_base, silver_base):
    """
    Xây dựng bảng fact_bureau từ Bronze layer:
    - Chuyển các cột DAYS_* (kiểu DOUBLE) thành INT trước khi tính ngày.
    - Tạo các cột date dựa trên ingest_date + DAYS_*.
    - Chọn FK và các measure cần thiết.
    """
    df = spark.read.format("delta").load(f"{bronze_base}/bureau")

    df_fact = (
        df
        # Chuyển ngày tương đối sang ngày tuyệt đối, ép kiểu DAYS_* về INT
        .withColumn(
            "credit_date",
            date_add(col("ingest_date"), col("DAYS_CREDIT").cast("int"))
        )
        .withColumn(
            "enddate_date",
            date_add(col("ingest_date"), col("DAYS_CREDIT_ENDDATE").cast("int"))
        )
        .withColumn(
            "close_date",
            date_add(col("ingest_date"), col("DAYS_ENDDATE_FACT").cast("int"))
        )
        .withColumn(
            "update_date",
            date_add(col("ingest_date"), col("DAYS_CREDIT_UPDATE").cast("int"))
        )
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            col("SK_ID_BUREAU").alias("sk_id_bureau"),
            col("CREDIT_ACTIVE").alias("credit_active"),
            col("CREDIT_CURRENCY").alias("credit_currency"),
            "credit_date",
            col("CREDIT_DAY_OVERDUE").alias("credit_day_overdue"),
            "enddate_date",
            "close_date",
            col("AMT_CREDIT_MAX_OVERDUE").alias("amt_credit_max_overdue"),
            col("CNT_CREDIT_PROLONG").alias("cnt_credit_prolong"),
            col("AMT_CREDIT_SUM").alias("amt_credit_sum"),
            col("AMT_CREDIT_SUM_DEBT").alias("amt_credit_sum_debt"),
            col("AMT_CREDIT_SUM_LIMIT").alias("amt_credit_sum_limit"),
            col("AMT_CREDIT_SUM_OVERDUE").alias("amt_credit_sum_overdue"),
            col("CREDIT_TYPE").alias("credit_type"),
            "update_date",
            col("AMT_ANNUITY").alias("amt_annuity")
        )
    )

    df_fact.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/fact_bureau")

def build_fact_bureau_balance(spark, bronze_base, silver_base):
    """
    Xây dựng bảng fact_bureau_balance từ Bronze layer:
      - balance_date: ingest_date + MONTHS_BALANCE (tháng)
      - status_code: nguyên giá trị STATUS (C, X, 0, 1, …)
      - status_desc: mô tả rõ ràng của mã trạng thái
      - FK: sk_id_bureau
    """
    # Đọc Bronze bureau_balance
    df = spark.read.format("delta").load(f"{bronze_base}/bureau_balance")

    df_fact = (
        df
        # Tính ngày balance tuyệt đối
        .withColumn(
            "balance_date",
            to_date(
                date_format(
                    add_months(col("ingest_date"), col("MONTHS_BALANCE").cast("int")),
                    "yyyy-MM-dd"
                )
            )
        )
        # Giữ nguyên code
        .withColumn("status_code", col("STATUS"))
        # Mô tả status
        .withColumn(
            "status_desc",
            when(col("STATUS") == "C", "closed")
            .when(col("STATUS") == "X", "unknown")
            .when(col("STATUS") == "0", "no_DPD")
            .when(col("STATUS") == "1", "DPD_1_30")
            .when(col("STATUS") == "2", "DPD_31_60")
            .when(col("STATUS") == "3", "DPD_61_90")
            .when(col("STATUS") == "4", "DPD_91_120")
            .when(col("STATUS") == "5", "DPD_120_plus")
            .otherwise("other")
        )
        # Chọn các cột cần thiết
        .select(
            col("SK_ID_BUREAU").alias("sk_id_bureau"),
            "balance_date",
            "status_code",
            "status_desc"
        )
    )

    # Ghi ra Delta Silver
    df_fact.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/fact_bureau_balance")

def build_fact_previous_application(spark, bronze_base, silver_base):
    """
    Xây dựng bảng fact_previous_application từ Bronze layer:
      - Chuyển các cột DAYS_* thành các date columns trên ingest_date
      - Lưu tất cả các trường cần thiết với FK và measures
    """
    df = spark.read.format("delta").load(f"{bronze_base}/previous_application")

    # Tạo các cột ngày tương đối thành date tuyệt đối
    df_fact = (
        df
        .withColumn(
            "application_date_id",
            date_format(col("ingest_date"), "yyyyMMdd").cast("int")
        )
        .withColumn(
            "decision_date",
            to_date(date_add(col("ingest_date"), col("DAYS_DECISION").cast("int")))
        )
        .withColumn(
            "first_drawing_date",
            to_date(date_add(col("ingest_date"), col("DAYS_FIRST_DRAWING").cast("int")))
        )
        .withColumn(
            "first_due_date",
            to_date(date_add(col("ingest_date"), col("DAYS_FIRST_DUE").cast("int")))
        )
        .withColumn(
            "last_due_1st_date",
            to_date(date_add(col("ingest_date"), col("DAYS_LAST_DUE_1ST_VERSION").cast("int")))
        )
        .withColumn(
            "last_due_date",
            to_date(date_add(col("ingest_date"), col("DAYS_LAST_DUE").cast("int")))
        )
        .withColumn(
            "termination_date",
            to_date(date_add(col("ingest_date"), col("DAYS_TERMINATION").cast("int")))
        )
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            col("SK_ID_PREV").alias("sk_id_prev"),
            "application_date_id",
            col("NAME_CONTRACT_TYPE").alias("name_contract_type"),
            col("AMT_ANNUITY").cast("double").alias("amt_annuity"),
            col("AMT_APPLICATION").cast("double").alias("amt_application"),
            col("AMT_CREDIT").cast("double").alias("amt_credit"),
            col("AMT_DOWN_PAYMENT").cast("double").alias("amt_down_payment"),
            col("AMT_GOODS_PRICE").cast("double").alias("amt_goods_price"),
            col("WEEKDAY_APPR_PROCESS_START").alias("weekday_appr_process_start"),
            col("HOUR_APPR_PROCESS_START").cast("int").alias("hour_appr_process_start"),
            # Flags
            when(col("FLAG_LAST_APPL_PER_CONTRACT") == "Y", 1).otherwise(0).alias("flag_last_appl_per_contract"),
            when(col("NFLAG_LAST_APPL_IN_DAY") == 1, 1).otherwise(0).alias("nflag_last_appl_in_day"),
            # Rates
            col("RATE_DOWN_PAYMENT").cast("double").alias("rate_down_payment"),
            col("RATE_INTEREST_PRIMARY").cast("double").alias("rate_interest_primary"),
            col("RATE_INTEREST_PRIVILEGED").cast("double").alias("rate_interest_privileged"),
            # Descriptors
            col("NAME_CASH_LOAN_PURPOSE").alias("name_cash_loan_purpose"),
            col("NAME_CONTRACT_STATUS").alias("name_contract_status"),
            "decision_date",
            col("NAME_PAYMENT_TYPE").alias("name_payment_type"),
            col("CODE_REJECT_REASON").alias("code_reject_reason"),
            col("NAME_TYPE_SUITE").alias("name_type_suite"),
            col("NAME_CLIENT_TYPE").alias("name_client_type"),
            col("NAME_GOODS_CATEGORY").alias("name_goods_category"),
            col("NAME_PORTFOLIO").alias("name_portfolio"),
            col("NAME_PRODUCT_TYPE").alias("name_product_type"),
            col("CHANNEL_TYPE").alias("channel_type"),
            col("SELLERPLACE_AREA").alias("sellerplace_area"),
            col("NAME_SELLER_INDUSTRY").alias("name_seller_industry"),
            col("CNT_PAYMENT").cast("int").alias("cnt_payment"),
            col("NAME_YIELD_GROUP").alias("name_yield_group"),
            col("PRODUCT_COMBINATION").alias("product_combination"),
            "first_drawing_date",
            "first_due_date",
            "last_due_1st_date",
            "last_due_date",
            "termination_date",
            when(col("NFLAG_INSURED_ON_APPROVAL") == 1, 1).otherwise(0).alias("nflag_insured_on_approval")
        )
    )

    # Ghi ra Delta Silver
    df_fact.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/fact_previous_application")

def build_fact_pos_cash_balance(spark, bronze_base, silver_base):
    """
    Xây dựng bảng fact_pos_cash_balance từ Bronze layer:
      - months_balance: giữ nguyên giá trị âm (relative month)
      - cnt_instalment, cnt_instalment_future: các chỉ số kỳ hạn
      - name_contract_status: trạng thái hợp đồng
      - sk_dpd, sk_dpd_def: các chỉ số DPD
    """
    df = spark.read.format("delta").load(f"{bronze_base}/POS_CASH_balance")

    df_fact = (
        df
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            col("SK_ID_PREV").alias("sk_id_prev"),
            col("MONTHS_BALANCE").cast("int").alias("months_balance"),
            col("CNT_INSTALMENT").cast("int").alias("cnt_instalment"),
            col("CNT_INSTALMENT_FUTURE").cast("int").alias("cnt_instalment_future"),
            col("NAME_CONTRACT_STATUS").alias("name_contract_status"),
            col("SK_DPD").cast("int").alias("sk_dpd"),
            col("SK_DPD_DEF").cast("int").alias("sk_dpd_def")
        )
    )

    df_fact.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/fact_pos_cash_balance")

def build_fact_credit_card_balance(spark, bronze_base, silver_base):
    """
    Xây dựng bảng fact_credit_card_balance từ Bronze layer:
      - months_balance: tháng tương đối (-1 = mới nhất)
      - các trường AMT_* (cast về double)
      - các trường CNT_* (cast về int)
      - name_contract_status, sk_dpd, sk_dpd_def
      - FK: sk_id_curr, sk_id_prev
    """
    # Đọc Bronze credit_card_balance
    df = spark.read.format("delta").load(f"{bronze_base}/credit_card_balance")

    df_fact = (
        df
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            col("SK_ID_PREV").alias("sk_id_prev"),
            col("MONTHS_BALANCE").cast("int").alias("months_balance"),
            # Amounts
            col("AMT_BALANCE").cast("double").alias("amt_balance"),
            col("AMT_CREDIT_LIMIT_ACTUAL").cast("double").alias("amt_credit_limit_actual"),
            col("AMT_DRAWINGS_ATM_CURRENT").cast("double").alias("amt_drawings_atm_current"),
            col("AMT_DRAWINGS_CURRENT").cast("double").alias("amt_drawings_current"),
            col("AMT_DRAWINGS_OTHER_CURRENT").cast("double").alias("amt_drawings_other_current"),
            col("AMT_DRAWINGS_POS_CURRENT").cast("double").alias("amt_drawings_pos_current"),
            col("AMT_INST_MIN_REGULARITY").cast("double").alias("amt_inst_min_regularity"),
            col("AMT_PAYMENT_CURRENT").cast("double").alias("amt_payment_current"),
            col("AMT_PAYMENT_TOTAL_CURRENT").cast("double").alias("amt_payment_total_current"),
            col("AMT_RECEIVABLE_PRINCIPAL").cast("double").alias("amt_receivable_principal"),
            col("AMT_RECIVABLE").cast("double").alias("amt_recivable"),
            col("AMT_TOTAL_RECEIVABLE").cast("double").alias("amt_total_receivable"),

            # Counts
            col("CNT_DRAWINGS_ATM_CURRENT").cast("int").alias("cnt_drawings_atm_current"),
            col("CNT_DRAWINGS_CURRENT").cast("int").alias("cnt_drawings_current"),
            col("CNT_DRAWINGS_OTHER_CURRENT").cast("int").alias("cnt_drawings_other_current"),
            col("CNT_DRAWINGS_POS_CURRENT").cast("int").alias("cnt_drawings_pos_current"),
            col("CNT_INSTALMENT_MATURE_CUM").cast("int").alias("cnt_instalment_mature_cum"),

            # Status & DPD
            col("NAME_CONTRACT_STATUS").alias("name_contract_status"),
            col("SK_DPD").cast("int").alias("sk_dpd"),
            col("SK_DPD_DEF").cast("int").alias("sk_dpd_def")
        )
    )

    # Ghi ra Delta Silver
    df_fact.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/fact_credit_card_balance")


def build_fact_installments_payments(spark, bronze_base, silver_base):
    """
    Xây dựng bảng fact_installments_payments từ Bronze layer:
      - FK: sk_id_curr, sk_id_prev
      - Các trường về kỳ hạn, ngày tương đối và số tiền:
        * num_instalment_version: phiên bản lịch trả
        * num_instalment_number: kỳ trả thứ mấy
        * days_instalment: số ngày (relative) đến ngày đến hạn
        * days_entry_payment: số ngày (relative) đến ngày đã thanh toán
        * amt_instalment: số tiền phải trả
        * amt_payment: số tiền thực tế đã trả
    """
    df = spark.read.format("delta").load(f"{bronze_base}/installments_payments")

    df_fact = (
        df
        .select(
            col("SK_ID_CURR").alias("sk_id_curr"),
            col("SK_ID_PREV").alias("sk_id_prev"),
            col("NUM_INSTALMENT_VERSION").cast("int").alias("num_instalment_version"),
            col("NUM_INSTALMENT_NUMBER").cast("int").alias("num_instalment_number"),
            col("DAYS_INSTALMENT").cast("int").alias("days_instalment"),
            col("DAYS_ENTRY_PAYMENT").cast("int").alias("days_entry_payment"),
            col("AMT_INSTALMENT").cast("double").alias("amt_instalment"),
            col("AMT_PAYMENT").cast("double").alias("amt_payment")
        )
    )

    df_fact.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", True) \
        .save(f"{silver_base}/fact_installments_payments")

if __name__ == "__main__":
    spark = create_spark_session()

    BRONZE_BASE = "s3a://lakehouse/bronze"
    SILVER_BASE = "s3a://lakehouse/silver"

    # build_dim_date(spark, BRONZE_BASE, SILVER_BASE)
    # build_dim_client(spark, BRONZE_BASE, SILVER_BASE)
    # build_dim_contract(spark, BRONZE_BASE, SILVER_BASE)
    # build_dim_employment(spark, BRONZE_BASE, SILVER_BASE)
    # build_dim_ext_source(spark, BRONZE_BASE, SILVER_BASE)
    # build_dim_document_flags(spark, BRONZE_BASE, SILVER_BASE)
    # build_dim_cb_requests(spark, BRONZE_BASE, SILVER_BASE)

    # build_fact_application(spark, BRONZE_BASE, SILVER_BASE)
    # build_fact_bureau(spark, BRONZE_BASE, SILVER_BASE)
    # build_fact_bureau_balance(spark, BRONZE_BASE, SILVER_BASE)
    # build_fact_previous_application(spark, BRONZE_BASE, SILVER_BASE)
    # build_fact_pos_cash_balance(spark, BRONZE_BASE, SILVER_BASE)
    # build_fact_credit_card_balance(spark, BRONZE_BASE, SILVER_BASE)
    # build_fact_installments_payments(spark, BRONZE_BASE, SILVER_BASE)
    spark.stop()


