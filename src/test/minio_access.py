import boto3

# Create a session using your MinIO credentials
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',  # MinIO endpoint
    aws_access_key_id='minioadmin',    # MinIO access key
    aws_secret_access_key='minioadmin123'  # MinIO secret key
)

# Check if the object exists
try:
    response = s3_client.head_object(Bucket='mlflow', Key='0/dfa4b5745cc54c0baec7f540226d57b1/artifacts/Random_Forest_n_100/model.pkl')
    print("Object exists:", response)
except Exception as e:
    print("Error:", e)