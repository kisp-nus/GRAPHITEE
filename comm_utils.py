"""Implement communication utilities for GNN models"""

from typing import Dict, List
from multiprocessing.pool import ThreadPool
import time
import json

import os

import torch.distributed as dist
import torch
import torch.nn as nn
import numpy as np

import requests
import boto3
import subprocess
import socket



from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from performance import PerformanceStore

def get_identity_document():
    """
    Get identity document for current EC2 Host
    """
    r = requests.get(
        "http://169.254.169.254/latest/dynamic/instance-identity/document")
    return r


def get_region(identity):
    """
    Get account of current instance identity
    """
    region = identity.json()["region"]
    return region


def get_account(identity):
    """
    Get account of current instance identity
    """
    account = identity.json()["accountId"]
    return account

def get_cid():
    """
    Determine CID of Current Enclave
    """
    proc = subprocess.Popen(["/bin/nitro-cli", "describe-enclaves"],
                            stdout=subprocess.PIPE)
    output = json.loads(proc.communicate()[0].decode())
    enclave_cid = output[0]["EnclaveCID"]
    return enclave_cid


# def set_identity():
#     identity = get_identity_document()
#     region = get_region(identity)
#     account = get_account(identity)
#     return region, account

# REGION, ACCOUNT = set_identity()
    
class MultiThreadReducerCentralized:
    """Multi-threaded reducer for aggregating gradients in a centralized manner"""
    

    def prepare_enclave_request(self, ciphertext):
        """
        Get the AWS credential from EC2 instance metadata. Only needed for KMS encrypted
        data. (so that the enclave has the credentials to decrypt it)
        """

        credential = {
            'access_key_id': self.access_key_id,
            'secret_access_key': self.secret_access_key,
            'token': self.token,
            'region': self.region,
            'ciphertext': ciphertext
        }

        return credential

    def __init__(self, model, device="cuda", sleep_time=0.1, pubkey=None, perf_store=None, measure_comm=False):
        self.model = model
        self._group = {}
        self.thread_pool = None
        self.sleep_time = sleep_time

        self.measure_comm = measure_comm
        self.comm_vol_store = perf_store
        self.pubkey = pubkey
        
        self.first_run = True
        # Only if KMS is used        
        # r = requests.get(
        #     "http://169.254.169.254/latest/meta-data/iam/security-credentials/")
        # instance_profile_name = r.text

        # r = requests.get(
        #     "http://169.254.169.254/latest/meta-data/iam/security-credentials/%s" %
        #     instance_profile_name)
        # response = r.json()
        # self.access_key_id = response['AccessKeyId']
        # self.secret_access_key = response['SecretAccessKey']
        # self.token = response['Token']
        # self.region = REGION
        
        user_keys = None
        
        # TODO (for local tests)
        with open("private_key.pem", "rb") as private_file:
            self.private_key = serialization.load_pem_private_key(
                private_file.read(),
                password=None,
                backend=default_backend()
            )
            
            
    def master_aggregate_plain(self, cfg, layer, perf_stores):
        """Aggregate plantext gradients on the master node"""
        world_size = cfg.num_partitions + 1
        
        all_grads = []
        num_elements = []
        shapes = []
        
        perf_store = PerformanceStore()
        perf_stores.append(perf_store)

    
        # with torch.no_grad():
        for _, (name, param) in enumerate(self.model.named_parameters()):
            all_grads.append(torch.zeros_like(param).cpu().numpy())
            shapes.append(param.shape)
            num_elements.append(param.numel())
    
        all_grads_array = np.concatenate([grad.flatten() for grad in all_grads])
        target_tensor = torch.tensor(np.frombuffer(all_grads_array.tobytes(), dtype=np.float32))
        
        
        for round in range(cfg.num_rounds[layer]):
            indexes_grads = []
            for _ in range(1, world_size):  # Exclude master itself
                # Placeholder tensor for gathering encrypted gradients
                i_g = torch.zeros(target_tensor.shape[0], dtype=torch.float32)  
                
                # Adjust the size as needed
                dist.recv(i_g, src=_, tag=round)
                indexes_grads.append(i_g.numpy())
                
            aggr_time_s = time.time()
              
            gradients = []
            for index, g_i in zip(range(len(indexes_grads)), indexes_grads):
                # Extract metadata
                num_params = int.from_bytes(g_i[:1], 'little')
                metadata_size = 1 + num_params# * 4
                indexes = np.frombuffer(g_i[1:metadata_size], dtype=np.int32)
                
                gradients_size = 0
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i in indexes:
                        gradients_size += param.numel()
                
                # print("grad size = ", gradients_size)
                offset = metadata_size + gradients_size
                grads_bytes = g_i[metadata_size:offset]

                grad = np.frombuffer(grads_bytes, dtype=np.float32)
                gradients.append(grad)
                
            # Aggregate decrypted gradients
            aggregated_grads = np.sum(gradients, axis=0) / world_size

            # Convert aggregated gradients back to tensor and set to param.grad
            start = 0
            # i = 0
            index = 0
            for param in self.model.parameters():
                if index in indexes:
                    grad_array = aggregated_grads[start:start + param.numel()].reshape(param.shape)
                    param.grad = torch.tensor(grad_array).view(param.shape)
                    start += param.numel()
                    # i += 1
                index+=1


            # Broadcast the aggregated gradients from rank 0 to all workers
            for (name, param) in self.model.named_parameters():
                if param.grad is not None:
                    dist.broadcast(param.grad, src=0)
                    
            aggr_time = time.time() - aggr_time_s
            if self.first_run:
                self.first_run = False
            else:
                perf_store.add_grad_reduce_time(aggr_time)
            
            
    def master_aggregate_gradients(self, cfg, layer, perf_stores):
        """Aggregate encrypted gradients directly on the master node"""
        world_size = cfg.num_partitions + 1
        


        all_grads = []
        num_elements = []
        shapes = []
        
        perf_store = PerformanceStore()
        perf_stores.append(perf_store)
        
        dummy_iv = os.urandom(16)
        dummy_key = os.urandom(32)
    
        # with torch.no_grad():
        for _, (name, param) in enumerate(self.model.named_parameters()):
            all_grads.append(torch.zeros_like(param).cpu().numpy())
            shapes.append(param.shape)
            num_elements.append(param.numel())
    
        all_grads_array = np.concatenate([grad.flatten() for grad in all_grads])
        target_tensor = torch.tensor(np.frombuffer(all_grads_array.tobytes() + dummy_iv + dummy_key, dtype=np.float32))
        
        # if self.user_keys is None:
        self.user_keys = [None for _ in range(1, cfg.num_partitions + 1)]
        
        for round in range(cfg.num_rounds[layer]):
            aggr_time_s = time.time()
            
            encrypted_list = []
            for _ in range(1, world_size):  # Exclude master itself
                # Placeholder tensor for gathering encrypted gradients
                encrypted_grads_iv_key = torch.zeros(target_tensor.shape[0], dtype=torch.float32)  
                
                # Adjust the size as needed
                dist.recv(encrypted_grads_iv_key, src=_, tag=round)
                encrypted_list.append(encrypted_grads_iv_key.numpy())
                
              
            decrypted_list = []
            for index, encrypted in zip(range(len(encrypted_list)), encrypted_list):
                # Extract metadata
                num_params = int.from_bytes(encrypted[:1], 'little')
                metadata_size = 1 + num_params# * 4
                indexes = np.frombuffer(encrypted[1:metadata_size], dtype=np.int32)
                
                gradients_size = 0
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i in indexes:
                        gradients_size += param.numel()
                
                # print("grad size = ", gradients_size)
                offset = metadata_size + gradients_size
                iv = encrypted[offset:offset+4].tobytes()
                encrypted_key = encrypted[offset+4:offset+68].tobytes()
                encrypted_grads = encrypted[metadata_size:offset]

                # Decrypt symmetric key
                if self.user_keys[index] is None:
                    priv_key_time_s = time.time()
                    self.user_keys[index] = self.private_key.decrypt(
                        encrypted_key,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )   
                    )
                    priv_key_time = time.time() - priv_key_time_s
                    print("Private key decryption time: ", priv_key_time)
                symmetric_key = self.user_keys[index]
                
                # Decrypt gradients
                decrypt_time_s = time.time()
                cipher = Cipher(algorithms.AES(symmetric_key), modes.CFB(iv), backend=default_backend())
                decryptor = cipher.decryptor()
                decrypted_bytes = decryptor.update(encrypted_grads.tobytes()) + decryptor.finalize()
                decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
                decrypted_list.append(decrypted_array)
                perf_store.add_grad_decryption(time.time() - decrypt_time_s)

            # Aggregate decrypted gradients
            aggregated_grads = np.sum(decrypted_list, axis=0) / world_size

            # Convert aggregated gradients back to tensor and set to param.grad
            start = 0
            # i = 0
            index = 0
            for param in self.model.parameters():
                if index in indexes:
                    grad_array = aggregated_grads[start:start + param.numel()].reshape(param.shape)
                    param.grad = torch.tensor(grad_array).view(param.shape)
                    start += param.numel()
                    # i += 1
                index+=1


            # Broadcast the aggregated gradients from rank 0 to all workers
            for (name, param) in self.model.named_parameters():
                if param.grad is not None:
                    dist.broadcast(param.grad, src=0)
                    
            aggr_time = time.time() - aggr_time_s
            perf_store.add_grad_reduce_time(aggr_time)
            
    def master_aggregate_enclave(self, cfg, layer, perf_stores):
        """Aggregate the gradients on the master node"""
        world_size = cfg.num_partitions + 1

                
        all_grads = []
        num_elements = []
        shapes = []
        
        perf_store = PerformanceStore()
        perf_stores.append(perf_store)
        
        dummy_iv = os.urandom(16)
        dummy_key = os.urandom(32)
    
        # with torch.no_grad():
        for _, (name, param) in enumerate(self.model.named_parameters()):
            all_grads.append(torch.zeros_like(param).cpu().numpy())
            shapes.append(param.shape)
            num_elements.append(param.numel())
    
        all_grads_array = np.concatenate([grad.flatten() for grad in all_grads])
        target_tensor = torch.tensor(np.frombuffer(all_grads_array.tobytes() + dummy_iv + dummy_key, dtype=np.float32))
        
        # if self.user_keys is None:
        self.user_keys = [None for _ in range(1, cfg.num_partitions + 1)]
        
        s = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        cid = get_cid()
        port = 5003
        
        # If debugging locaaly:
        # cid = "127.0.0.1"
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        s.connect((cid, port))
        send_metadata = True

            
        for round in range(cfg.num_rounds[layer]):
            aggr_time_s = time.time()
            # print("Round: ", round)
            
            encrypted_list = []
            for _ in range(1, world_size):  # Exclude master itself
                # Placeholder tensor for gathering encrypted gradients
                encrypted_grads_iv_key = torch.zeros(target_tensor.shape[0], dtype=torch.float32)  
                
                # Adjust the size as needed
                dist.recv(encrypted_grads_iv_key, src=_, tag=round)
                encrypted_list.append(encrypted_grads_iv_key.numpy().tobytes())

                
                
            # First round send all metadata: # of clients and buffer sizes
            if send_metadata:
                # Just for convenience of gradient indexes 
                for index, encrypted in zip(range(len(encrypted_list)), encrypted_list):
                    # Extract metadata
                    num_params = int.from_bytes(encrypted[:1], 'little')
                    metadata_size = 1 + num_params * 4
                    indexes = np.frombuffer(encrypted[1:metadata_size], dtype=np.dtype('>i4'))
                    
                gradients_size = 0
                for i in range(len(num_elements)):
                    if i in indexes:
                        gradients_size += num_elements[i]
                metadata = {
                    'num_clients': cfg.num_partitions,
                    'buffer_size': len(encrypted_list[0]),
                    'grad_size': gradients_size,
                    'total_rounds': cfg.num_rounds[layer]
                }
                
                metadata = str.encode(json.dumps(metadata))
                s.send(metadata)
                send_metadata = False
            
            # print("Sending the tensors")
            for gradient_buffer in encrypted_list:
                s.sendall(gradient_buffer)
                
                
            buffer_size = gradients_size * 4
            r = s.recv(buffer_size)
            
            # If we didn't get all the data, keep receiving
            while len(r) < buffer_size:
                remaining = (buffer_size) - len(r)
                chunk = s.recv(remaining)
                if not chunk:
                    raise ConnectionError("Connection closed before receiving full gradient")
                r += chunk
            
            # parsed = json.loads(r)
            decrypted_gradient = np.frombuffer(r, dtype=np.float32)

            # Convert aggregated gradients back to tensor and set to param.grad
            start = 0
            # i = 0
            index = 0
            for param in self.model.parameters():
                if index in indexes:
                    grad_array = decrypted_gradient[start:start + param.numel()].reshape(param.shape)
                    param.grad = torch.tensor(grad_array).view(param.shape)
                    start += param.numel()
                    # i += 1
                index+=1


            # Broadcast the aggregated gradients from rank 0 to all workers
            for (name, param) in self.model.named_parameters():
                if param.grad is not None:
                    dist.broadcast(param.grad, src=0)
                    
            aggr_time = time.time() - aggr_time_s
            perf_store.add_grad_reduce_time(aggr_time)
    
    def master_aggregate_users(self, cfg):
        """Aggregate the user embeddings on the master node"""
        world_size = cfg.num_partitions + 1
        
        all_users = []
        num_elements = []
        shapes = []
        
        for _, (name, param) in enumerate(self.model.named_parameters()):
            all_users.append(torch.zeros_like(param).cpu().numpy())
            shapes.append(param.shape)
            num_elements.append(param.numel())
    
        all_users_array = np.concatenate([user.flatten() for user in all_users])
        # compute size of history + user embedding tensor
        target_tensor = torch.tensor(np.frombuffer(all_users_array.tobytes(), dtype=np.float32))
        
        encrypted_list = []
        decrypted_list = []
        for worker_rank in range(1, world_size):
            # 1. Receive total tensor sizes.
            size_tensor = torch.empty(18, dtype=torch.float32)  # Adjust the size based on the actual data being sent
            dist.recv(size_tensor, src=worker_rank, tag=1_000_000)
            iv_size = 16  # size of the IV for AES
            encrypted_size = size_tensor[iv_size:].numpy().tobytes()
            iv = size_tensor[:iv_size].numpy().tobytes()
            
            if self.symmetric_key is None:
                self.symmetric_key = os.urandom(32)  # Temporary assignment, should be shared securely beforehand
            
            cipher = Cipher(algorithms.AES(self.symmetric_key), modes.CFB(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_sizes = decryptor.update(encrypted_size) + decryptor.finalize()
            tensor_sizes = np.frombuffer(decrypted_sizes, dtype=np.int32)
            
            total_size = tensor_sizes[0] + tensor_sizes[1] + tensor_sizes[2]  # Sum up the sizes of all parts
            
            # 2. Receive encrypted tensors
            encrypted_tensor = torch.empty(total_size + iv_size, dtype=torch.float32)
            dist.recv(encrypted_tensor, src=worker_rank, tag=1_000_001)
            iv = encrypted_tensor[:iv_size].numpy().tobytes()
            encrypted_data = encrypted_tensor[iv_size:].numpy().tobytes()
            
            cipher = Cipher(algorithms.AES(self.symmetric_key), modes.CFB(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            all_tensors = np.frombuffer(decrypted_data, dtype=np.float32)
            
            id_map_keys_size = tensor_sizes[0]
            global_news_ids_size = tensor_sizes[1]
            user_features_size = tensor_sizes[2]
            
            id_map_keys = all_tensors[:id_map_keys_size]
            global_news_ids = all_tensors[id_map_keys_size:id_map_keys_size + global_news_ids_size]
            user_features = all_tensors[id_map_keys_size + global_news_ids_size:]
            
            # OLD CODE
            # Placeholder tensor for gathering encrypted user data
            encrypted_data = torch.zeros(target_tensor.shape[0], dtype=torch.float32)  
            
            # Adjust the size as needed
            dist.recv(encrypted_data, src=_, tag=round)
            encrypted_list.append(encrypted_data.numpy())
            
        for index, encrypted in zip(range(len(encrypted_list)), encrypted_list):
            # Extract metadata
            num_params = int.from_bytes(encrypted[:1], 'little')
            metadata_size = 1 + num_params
            
            symmetric_key = self.user_keys[index]
            
            iv = encrypted[:4].tobytes()
            
            history_size = 0 # TODO
            embedding_size = 0 # TODO
            
            # Decrypt vectors
            cipher = Cipher(algorithms.AES(symmetric_key), modes.CFB(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_bytes = decryptor.update(encrypted.tobytes()) + decryptor.finalize()
            history = np.frombuffer(decrypted_bytes[4: 4 + history_size])
            embedding = np.frombuffer(decrypted_bytes[4 + history_size: 4 + history_size + embedding_size], dtype=np.flot32)
            
            
            for item in history:
                # TODO item aggregation
                pass
            
            # decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
            # decrypted_list.append(decrypted_array)



def extract_node_type(name, node_types):
    tokens = name.split(".")
    for type in node_types:
        if type in tokens:
            return type
    return "all"

def aggregate_metrics(metrics: Dict):
    """Aggregate the metrics across workers

    Parameters
    ----------
    metrics : Dict
        Metrics to aggregate

    Returns
    -------
    Dict
        Aggregated metrics
    """

    for k, v in metrics.items():
        t = torch.tensor(v).cuda()
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=False)
        metrics[k] = t.item() 

    return metrics

def sync_model(model: nn.Module):
    """Sync the model across workers

    Parameters
    ----------
    model : nn.Module
        Model to sync

    Returns
    -------
    nn.Module
        Synced model
    """

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def get_comm_size_param(param):
    """Get the communication size of a parameter

    Parameters
    ----------
    param : torch.Tensor
        Parameter

    Returns
    -------
    int
        Communication size of the parameter
    """

    return param.numel() * param.element_size()
