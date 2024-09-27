import boto3
import os 
import argparse


def download_directory_from_s3(bucket, s3_directory, local_directory):
    s3 = boto3.client('s3')
    
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=s3_directory):
        if 'Contents' in result:
            for file in result['Contents']:
                s3_path = file['Key']
                if not s3_path.endswith('/'):  # Skip directories
                    local_path = os.path.join(local_directory, os.path.relpath(s3_path, s3_directory))
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    print(local_path)
                    print(os.path.exists(local_path))
                    print(f"Downloading {s3_path} to {local_path}")
                    s3.download_file(bucket, s3_path, local_path)
                    print(os.path.exists(local_path))

# Download a file
# bucket_name = 'mind-central'
# file_name = 'datasets/info_small.json'
# s3.download_file(bucket_name, file_name, "test.json")


# Usage
bucket_name = 'mind-central'
s3_dir = 'datasets'
local_dir = './datasets'

# download_directory_from_s3(bucket_name, s3_dir, local_dir)


def main():
    parser = argparse.ArgumentParser(description='Download a directory from S3')
    parser.add_argument('--bucket_name', type=str, default="mind-central", help='Name of the S3 bucket')
    parser.add_argument('--s3_dir', type=str, default='datasets', help='S3 directory to download from. To download partition 0, should be "partition_0"')
    parser.add_argument('--local_dir', type=str, default='./datasets', help='Local directory to download to')
    
    args = parser.parse_args()

    download_directory_from_s3(args.bucket_name, args.s3_dir, args.local_dir)

if __name__ == "__main__":
    main()