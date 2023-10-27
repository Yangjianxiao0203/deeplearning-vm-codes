import sagemaker
import boto3

sm_boto3 = boto3.client('sagemaker') # create a SageMaker client
sm_session = sagemaker.Session() # create a SageMaker session
region = boto3.Session().region_name # set the region of the instance
bucket = "deeplearningbucket" # set my bucket name

print("SageMaker client: {}, region: {}, bucket: {}".format(sm_boto3,region,bucket))

# send data to S3. SageMaker will take training data from s3
sk_prefix = "data"
trainpath = sm_session.upload_data(path="data/data/dataset.csv", bucket=bucket, key_prefix=sk_prefix)
vocab = sm_session.upload_data(path="data/data/chars.txt", bucket=bucket, key_prefix=sk_prefix)