# Running GRAPHITEE on AWS (work-in-progress)


The instructions below explain how to run GRAPHITEE in a distributed setup. First, please make sure to use instances that support AWS Nitro (with Amazon Linux 2023 AMI), ideally with at least 4 vCPUs and 4 GB of RAM for the controller.

## Setting up the controller

Pull the aws controller branch and install the required libraries from the installation file. Prepare the instance to use the enclave by running the following:

```
# Initialization (see [here](https://catalog.workshops.aws/nitro-enclaves/en-US/0-getting-started/prerequisites#install-and-configure-nitro-enclaves-cli-and-tools))
sudo amazon-linux-extras install aws-nitro-enclaves-cli -y
sudo yum install aws-nitro-enclaves-cli-devel -y
sudo usermod -aG ne $USER
sudo usermod -aG docker $USER

sudo systemctl start nitro-enclaves-allocator.service && sudo systemctl enable nitro-enclaves-allocator.service
sudo systemctl start docker && sudo systemctl enable docker

# Set environment variable (if new machine type, check with ifconfig the if name)
export GLOO_SOCKET_IFNAME=ens5

# build the docker image
cd enclave
docker build . -t enclave --no-cache

# create enclave image
nitro-cli build-enclave --docker-uri enclave --output-file enclave.eif

# run enclave
nitro-cli run-enclave --cpu-count 2 --memory 2048 --enclave-cid 16 --eif-path enclave.eif --debug-mode 

```

You can then start the GRAPHITEE process normally. The controller address will be the instance's private ip. Make sure your instance's firewall does not block TCP traffic on the port you choose for training.

## Setting up the clients

Setting up a client is very straightforward. Pull the correct client branch and edit the controller address in the config file. The setup.sh script has everything to run the client directly; no further action is needed.
After the first run you can launch the main.py directly to avoid checking for the installed packages.

## Making partitions available in S3

The code supports having the data directly on an S3 server. If you decide to upload your partitions to an S3 server, you can use the "create_partitions" option in main.py. For large amounts of data or partitions, make sure to do this action from an AWS instance in the same region as the S3 server to avoid any bandwidth charge.
