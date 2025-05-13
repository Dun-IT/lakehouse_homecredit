from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 9),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='data_processing',
    default_args=default_args,
    description='Submit ingest script via BashOperator',
    schedule_interval='@daily',
    catchup=False,
    tags=['etl', 'ingest', 'lakehouse'],
) as dag:
    # 1) Bronze layer
    run_bronze_ingest = BashOperator(
        task_id='run_bronze_ingest',
        bash_command='bash /mnt/spark-apps/bronze/run_bronze_ingest.sh >> /opt/airflow/logs/bronze_ingest.log 2>&1',
    )

    # 2) Silver layer
    run_silver_ingest = BashOperator(
        task_id='run_silver_ingest',
        bash_command=
            'bash /mnt/spark-apps/silver/run_silver_ingest.sh >> /opt/airflow/logs/silver_ingest.log 2>&1',
    )

    # 3) Gold layer
    run_gold_ingest = BashOperator(
        task_id='run_gold_ingest',
        bash_command=
        'bash /mnt/spark-apps/gold/run_gold_ingest.sh >> /opt/airflow/logs/gold_ingest.log 2>&1',
    )


    run_bronze_ingest >> run_silver_ingest >> run_gold_ingest
