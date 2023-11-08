import boto3
import os

s3 = boto3.client('s3')

def upload_dataset():
    bucket_name = 'deeplearningbucket'
    s3_path = 'data/Rbert'
    local_path = '../triplet_data'
    #upload every file in local_path to s3_path
    for root, dirs, files in os.walk(local_path):
        for file in files:
            s3_file_path = os.path.join(root.replace(local_path, s3_path), file)
            local_file_path = os.path.join(root, file)
            s3.upload_file(local_file_path, bucket_name, s3_file_path)
            print(s3_file_path)

def upload_model():
    bucket_name = 'deeplearningbucket'
    s3_path = 'data/Rbert'
    local_path = './model_output'
        #upload every file in local_path to s3_path
    for root, dirs, files in os.walk(local_path):
        for file in files:
            s3_file_path = os.path.join(root.replace(local_path, s3_path), file)
            local_file_path = os.path.join(root, file)
            s3.upload_file(local_file_path, bucket_name, s3_file_path)
            print(s3_file_path)

if __name__ =='__main__':
    # upload_dataset()
    upload_model()
    print('upload finished! ')