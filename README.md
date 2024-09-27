# Retexo: Scalable Neural Network Training over Distributed Graphs

Graph data can be large and is often stored distributed across many machines (cluster machines, edge servers, or mobile devices). Distributed training of graph neural networks (GNNs) over data partitioned graphs is necessary, not just because of efficiency reasons, but because of compliance with data residency or privacy laws. However, existing ways of training GNNs over distributed infrastructure are extremely costly in terms of network communication. We present a framework called Retexo which eliminates the severe communication bottlenecks in distributed GNN training while respecting any given data partitioning configuration. We offer a new procedure to train popular GNN architectures distributively. Retexo can achieve **1-2 orders of magnitude reduction in network data costs** compared to standard training while retaining accuracy. Retexo can also train GNNs **2X faster than state-of-the-art baselines** for distributed graphs. Further, Retexo scales gracefully as the level of decentralization increases and can be deployed in mobile environments too.

**Lazy message-passing:** The key idea to address the communication bottlenecks is to train GNNs layer-by-layer instead of optimizing parameters of all layers together. Specifically, each GNN layer is trained independently of the other layers and sequentially after training all of its previous layers. The inputs to the first layer are obtained after one message-passing round over raw features. After the first layer is trained the subsequent layers can be trained by using the representations obtained from the last trained layer as inputs. This gives an effect of delaying a message-passing until completely training all previous layers of the GNN. Hence, we call this greedy layer-by-layer optimization procedure to train GNNs as *Lazy message-passing*. 

**Paper:** New version to be public, Previous version is on [arXiv](https://arxiv.org/abs/2302.13053).

**Authors:** Aashish Kolluri* and Sarthak Choudhary* and Bryan Hooi and Prateek Saxena\
(* Equal contribution.)

## Environment Setup
Retexo is built and tested using the following libraries.

 - python 3.8
 - CUDA 11.3
 - [PyTorch 1.12](https://github.com/pytorch/pytorch) ([wheel](https://download.pytorch.org/whl/cu113))
 - [PyG 2.4.0](https://pytorch-geometric.readthedocs.io/en/latest/), [PyTorch Sparse 0.6.16](https://github.com/rusty1s/pytorch_sparse), and [PyTorch Scatter 2.1.0](https://github.com/rusty1s/pytorch_scatter) ([wheels](https://data.pyg.org/whl/torch-1.12.1%2Bcu113.html))
 - [DGL 1.1.0](https://github.com/dmlc/dgl) ([wheel](https://data.dgl.ai/wheels/cu113/repo.html))
 - [OGB](https://github.com/snap-stanford/ogb)
 - [hydra](https://github.com/facebookresearch/hydra)
 - [torchmetrics](https://github.com/Lightning-AI/torchmetrics)

To setup the enviromnent, we recommend using Anaconda ([installation](https://www.anaconda.com/download)) as follows.

```
conda create -n retexo_py3.8 python=3.8
conda activate retexo_py3.8
```

The dependencies can be installed within the conda environment using the commands given in ```commands_cu11.3.txt``` file.

## Running Retexo on a Single Node (Machine)

To train GNNs using Retexo, one must first have the partitions of the graph data. Retexo can be used to partition the graph data as well before training.

### Data Paritioning
The following command can be used to partition a graph beforehand.

```bash
python main.py app=partition_data dataset_name=reddit num_partitions=4
```

This downloads and partitions the Reddit dataset. It stores the partitions in to the path specified by the ```partition_dir``` option.
```bash
ls ../retexo-datasets/partitions/reddit-random-vol-4/
# meta.json  part0  part1  part2  part3  reddit-random-vol-4.json
```

### Training
Retexo uses one worker per partition to train GNNs. Therefore, if there are 4 partitions then Retexo spawns 4 workers (processes), one per partition. Retexo can be run using arguments provided in the configuration files in ```conf``` directory as follows. The default configuration files provided can be used to train GNNs on a randomly partitioned Reddit dataset for node classification. The following commands can be used to train on a single machine using 4 workers (data partitions).

```bash
# to train a RetexoGCN on Reddit,
python main.py --config-name node_classification_gcn app=train

# RetexoSAGE,
python main.py --config-name node_classification_sage app=train

# RetexoGAT,
python main.py --config-name node_classification_gat app=train
```

To change the configuration such as dataset or hyperparameters, specify them as part of the command line arguments.

```bash
# to train a RetexoGCN on Products,
python main.py --config-name node_classification_gcn app=train dataset_name=ogbn-products

# use "+" to introduce new arguments. For instance, to train with SGD optimizer,
python main.py --config-name node_classification_gcn app=train dataset_name=reddit optimizer._target_=torch.optim.SGD +optimizer.momentum=0.9

# to measure the network data volume during training, use the measure_dv option
python main.py --config-name node_classification_gcn app=train measure_dv=true

# to manually add network latencies between workers, use the sleep_time option (in seconds)
python main.py --config-name node_classification_gcn app=train sleep_time 0.1
```

## Running Retexo on Multiple Nodes
Retexo can also be trained with data partitions located across physically seperated nodes that are connected to each other over network. Each node can spawn multiple workers depending on number of partitions available on that node. For instance, if 8 partitions of a graph are split equally among 2 nodes then each node spawns 4 workers one per partition, thus, in total there will be 8 workers. Retexo currently supports the gloo backend right now for inter-worker communication. Note that the partitions may have to be manually copied in to every node after partitioning the graph in one node.

To train Retexo on multiple nodes, the same training command will have to be run on each node with the same master address and the master port with unique node rank. For example,

```bash
# to train on 2 nodes with 2 partitions per node, on node 0 with IP address [IP]
python main.py --config-name node_classification_gcn app=train num_partitions=4 parts_per_node=2 node_rank=0 distributed.master_addr=[IP] distributed.master_port=10011

# on node 1,
python main.py --config-name node_classification_gcn app=train num_partitions=4 parts_per_node=2 node_rank=1 distributed.master_addr=[IP] distributed.master_port=10011
```

*Note that currently the pytorch distributed requires all ports on one node to be accessible to the application being run on the other nodes.*

## Results
We compare the effiecieny of training using Retexo against strandard training and boundary node sampling (BNS) a state-of-the-art baseline for efficient distributed GNN training.

### Accuracy vs Network Data Volume
The following figure compares the accuracy achieved vs the total data volume that was communicated over the network during training on two popular benchmarks for node classification.

<img title="a title" alt="Alt text" src="/img/img4.png">

### Accuracy
GNNs trained with Retexo achieve similar or better performance than standard training on many popular benchmarks for node classification.

<img title="a title" alt="Alt text" src="/img/img3.png">

### Contact
If you want to contribute to this project or have any queries please contact one of us.

Aashish Kolluri: aashish7@comp.nus.edu.sg, aashishkolluri6@gmail.com

Sarthak Choudhary: csarthak76@gmail.com

Bryan Hooi: dcsbhk@nus.edu.sg

Prateek Saxena: prateeks@comp.nus.edu.sg




