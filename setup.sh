#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$1" == "true" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 <run_download_script> <world_size> <partition_ID>"
        exit 1
    fi
    run_download_script=true
    world_size=$2
    partition_ID=$3
elif [ "$1" == "false" ]; then
    run_download_script=false
else
    echo "Usage: $0 <run_download_script> <opt: world_size> <opt: partition_ID>"
    exit 1
fi

curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user


# dependencies
pip install -r requirements.txt
pip install --no-deps torchdata==0.7.0

# Set environment variable (if new machine type, check with ifconfig the if name)
export GLOO_SOCKET_IFNAME=ens5

# AWS CLI (this is dynamic and asks for access and secret keys)
if [ "$run_download_script" == "true" ]; then
    aws configure
    python3 download_datasets.py --s3_dir partitions/$world_size/$partition_ID/
fi