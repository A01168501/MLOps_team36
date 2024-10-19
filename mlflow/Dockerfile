# Use base python image
FROM python:3.10-slim

# Establish directory
WORKDIR /app

RUN apt-get update && apt-get install -y curl
RUN pip install mlflow[extras] psycopg2-binary boto3 cryptography pymysql

# Copy the rest of the code
COPY . .

# Expose the port where MLFlow serves the UI.
EXPOSE 5000

# Execute mlflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "postgresql://${PG_USER}:${PG_PASSWORD}@db:${PG_PORT}/${PG_DATABASE}", "--serve-artifacts", "--artifacts-destination", "s3://${MLFLOW_BUCKET_NAME}"