#!/usr/bin/env bash

# Script để submit job Silver layer ingestion lên Spark cluster Docker

MASTER_CONTAINER="spark-apps-master"
# URL Spark Master
MASTER_URL="spark://spark-apps-master:7077"

MINIO_ENDPOINT="minio:9000"
MINIO_ACCESS_KEY="minio"
MINIO_SECRET_KEY="minio_admin"

SCRIPT_PATH="/opt/spark-apps/silver/silver_layer_ingest.py"

echo "==> Checking Python3 in container ${MASTER_CONTAINER}..."
docker exec -u root ${MASTER_CONTAINER} bash -lc "which python3 || (apt-get update && apt-get install -y python3)"

# Submit job sử dụng here-doc
echo "==> Submitting Silver ingestion job to Spark Master: ${MASTER_URL}"
docker exec -i ${MASTER_CONTAINER} bash <<EOF
cd /opt/spark-apps
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3
/opt/bitnami/spark/bin/spark-submit \
  --master ${MASTER_URL} \
  --deploy-mode client \
  --packages io.delta:delta-spark_2.12:3.3.0 \
  --jars \$(ls /opt/bitnami/spark/jars/*.jar | tr '\n' ',') \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  --conf spark.pyspark.python=python3 \
  --conf spark.pyspark.driver.python=python3 \
  --conf spark.hadoop.fs.s3a.endpoint=${MINIO_ENDPOINT} \
  --conf spark.hadoop.fs.s3a.access.key=${MINIO_ACCESS_KEY} \
  --conf spark.hadoop.fs.s3a.secret.key=${MINIO_SECRET_KEY} \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.rpc.message.maxSize=1024 \
  --conf spark.network.timeout=800s \
  --conf spark.executor.heartbeatInterval=60s \
  ${SCRIPT_PATH}
EOF

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "==> Job submitted successfully."
else
  echo "==> Job failed with exit code $EXIT_CODE. Check logs: docker logs ${MASTER_CONTAINER}" >&2
  exit $EXIT_CODE
fi
