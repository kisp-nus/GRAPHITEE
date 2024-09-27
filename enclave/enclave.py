# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: MIT-0
# // see https://github.com/aws-samples/aws-nitro-enclaves-workshop

# // Code here is a simple reference and requires a running enclave and a complete image.

import json
import base64
import socket
import subprocess
import numpy as np
import time

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


KMS_PROXY_PORT="8000"

TEST_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCP+IR1MsFDJsmS
T/ct0kajT+iJ5mkqZ8xn41mdUcSn+oJDIlEYy9oRx9eeH3ca4I3CPuaeFgwCHpKs
5TMGB7LkygHgokgyKNJa+mCA81V5LWRlDj41sT2PEy+8iuRZhyPmRa9jYL6a0Avr
dJ9MZ7NhwxH1rFrH+olT3py3SRHOLFl/Aiv9rM3vcJLfq4lYZkHICbknPFGjZHV4
PkKVWgbFlSYHeanJ46N3oWGOQeYTit9R5KecdwV/DeNup0efsDtQmhvB5dcALqwH
KfziAYb0K6TQEzfDcfdZhjSe18MJUYCrVeIRb4uMcH3mziSJRRI8KB05eSGJ1slG
DhveAPKXAgMBAAECggEAIDNTXOsnMp/AQctE26jGR8ydlApGx0dD/pBpALjDJVbw
P5ezT7p4YbWy0hjziL1kt2deKUmBEhBIegchbF3YczeDR/zD7QQYWGTbpLvICDxQ
0hFndJbZz+BYsvDVtfh13REE81M2DmYt5FHHN02SX3FD2RDdlRDCGlIV9yCOrPAE
i3TJZzNu2MaQo665j3eQvNMclZicpXV75Ea9J6eVDmI+eb3e4btbSZsyNUK/bxMU
AQz01Thbprmn6wMoE62ByrDNXsC0MCi2dcCmdRrCZJgHIUVL7BFLUrdMk1ub2Aen
eij7a/FpbRAZPTddvftNDCAF+LT3yrUo1G/LyvEqgQKBgQDAlsRd6WrfQ6RfSSGJ
ba5zME8BdCGMzTb5ssNu40x0ZggVa9tEXfRAs4Gq5T/Jx9B6SJbnt7vU4h2BqgZp
4B/aMMB0RxaQAh0fiBGEIrAzbAHWyQ2KGbL5lMt7wyDbMgstdaNU+1IVbp5dyV0o
zdM8p2tDdO6xxsCRYTUSV8MrdwKBgQC/X79qCmSL/a7n2ybzwgpFxCCwEpThBhBl
+ryTGoPU2FmrEUGKxLjpnOydVAHseujDNGPlpkGMokubbOar9XfDYNgU0UtdwGbf
D+4HQ9HgYT6F3/aaUrcAxbJRAC2KeuTzFhzrYWi2B5QwlDEdc2a+XN3JpwI8oW7O
5vezlZf54QKBgQCaFATmXWhzVtqaoReDq4x1+6A5uX9d9pCVFL/mZ1MzjK3K9Y9n
EwPm/7Yt01tFQ+c27fxNRmGv33db1XtsGTNijL5sSLN0YzyJjYL2BAqUGUUfYZrD
cewOYUyqp2IR++eVZxhVVPxGyUlKH1+41XK6g/b88QBGmoxVwz+CfVRX4QKBgE3a
QL99zNuabx4JbHY7mPw6xmV6mbBVTDSAZVier8TV4tyR34z5bJQ075ktRL8UXT6U
QJN7KC1zFLj8+3Y3HOqRqjYF0tgn3nGeRGIWN/pE9S93JhYv0hzxUBJdtSkhx6QQ
eeFTtkpfGO0OTFDD0qdclilj1KgfsDlhgqE/GR1hAoGBAJNsVHvt4n93PprlMing
Y7KAbnvRefV3QIm47P4B4ahwNnLOtJ+Ba8wf5NrspyQb0GIb5r6tvgTpwf3vSdaL
y65i1kFnYnIAUMKLn97J4+b9CFTn18J1eAvQHPZ4U7+7mCZFY17ofUMDc3aPqLdJ
sPFd7T9X5rB32ZVUPuxIv1yl
-----END PRIVATE KEY-----"""

def get_plaintext_kms(credentials):
    """
    Decrypts using KMS. Prepare inputs and invoke decrypt function.
    """

    # take all data from client
    access = credentials['access_key_id']
    secret = credentials['secret_access_key']
    token = credentials['token']
    ciphertext= credentials['ciphertext']
    region = credentials['region']
    print(f"ciphertext is {ciphertext}")
    tensors = [decrypt_cipher(access, secret, token, ct, credentials['region']) for ct in ciphertext]
    return tensors
    # creds = decrypt_cipher(access, secret, token, ciphertext, region)
    # return creds


def decrypt_cipher(access, secret, token, ciphertext, region):
    """
    use KMS Tool Enclave Cli to decrypt cipher text
    Look at https://github.com/aws/aws-nitro-enclaves-sdk-c/blob/main/bin/kmstool-enclave-cli/README.md
    for more info.
    """
    proc = subprocess.Popen(
    [
        "/app/kmstool_enclave_cli",
        "decrypt",
        "--region", region,
        "--aws-access-key-id", access,
        "--aws-secret-access-key", secret,
        "--aws-session-token", token,
        "--ciphertext", ciphertext,
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

    ret = proc.communicate()
    print(f"Decrypting one tensor {ciphertext}")

    if ret[0]:
        ret0 = proc.communicate()[0].decode()
        b64text = ret0.split(":")[1]
        plaintext = base64.b64decode(b64text)
        print(f"Plaintext is {plaintext}")

        tensor = np.frombuffer(plaintext, dtype=np.int32)
        return tensor
        # plaintext = base64.b64decode(b64text).decode('utf-8')
        # return plaintext
    else:
        print("KMS Error. Decryption Failed." + str(ret) + str(region) + str(access))
        return "KMS Error. Decryption Failed." + str(ret) + str(region) + str(access)#returning the full error stack when something fails


def main():

    private_key = serialization.load_pem_private_key(
        TEST_KEY_PEM.encode(),
        password=None,  
        backend=default_backend()
    )

    user_key = None
    s = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    
    
    cid = socket.VMADDR_CID_ANY
    port = 5003
    
    # If debugging locally uncomment:
    # cid = "127.0.0.1"
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    s.bind((cid, port))
    s.listen()
    print(f"Started server on port {port} and cid {cid}")
    num_layers = 3
    for _ in range(num_layers):
        aggregate_layer(private_key, user_key, s)
                


def aggregate_layer(private_key, user_key, s):

    world_size = 1
    total_rounds = 1
    current_round = 0
    
    metadata = None
    aggr_times = []
    decrypt_times = []
    
    c, addr = s.accept()
    print(f"Connection from {addr}")

    while True:
        decrypted_gradients = []
        if metadata is None:
            # Receive metadata on first round
            metadata_json = b""
            while True:
                chunk = c.recv(1)
                metadata_json += chunk
                try:
                    metadata = json.loads(metadata_json.decode())
                    print("Received metadata:", metadata)
                    world_size = metadata['num_clients']
                    buffer_size = metadata['buffer_size']
                    gradients_size = metadata['grad_size']
                    total_rounds = metadata['total_rounds']
                    break
                except json.JSONDecodeError:
                    # If we can't decode yet, we need more data
                    continue

        
        # Receive gradients
        if user_key is not None:
            aggr_s = time.time()
        else:
            aggr_s = None
        encrypted_gradients = []
        for _ in range(world_size):
            gradient_buffer = c.recv(buffer_size)
            
            # If we didn't get all the data, keep receiving
            while len(gradient_buffer) < buffer_size:
                remaining = buffer_size - len(gradient_buffer)
                chunk = c.recv(remaining)
                if not chunk:
                    raise ConnectionError("Connection closed before receiving full gradient")
                gradient_buffer += chunk
            
            encrypted_gradients.append(gradient_buffer)
        
        # print("Received the tensors")
        # Extract metadata
        num_params = int.from_bytes(gradient_buffer[:1], 'little')
        metadata_size = 1 + num_params# * 4
        
        for encrypted in encrypted_gradients:
            offset = int(metadata_size + gradients_size)
            gradient = np.frombuffer(encrypted, dtype=np.float32)
            iv = gradient[offset:offset+4].tobytes()
            encrypted_key = gradient[offset+4:offset+68].tobytes()
            encrypted_grads = gradient[metadata_size:offset]
            
            if user_key is None:
                priv_key_time_s = time.time()
                user_key = private_key.decrypt(
                    encrypted_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )   
                )
                priv_key_time = time.time() - priv_key_time_s
                print("Private key decryption time: ", priv_key_time)
            symmetric_key = user_key
            
            decrypt_time_s = time.time()
            cipher = Cipher(algorithms.AES(symmetric_key), modes.CFB(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_bytes = decryptor.update(encrypted_grads.tobytes()) + decryptor.finalize()
            decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
            decrypted_gradients.append(decrypted_array)
            decrypt_time = time.time() - decrypt_time_s
            decrypt_times.append(decrypt_time)

        tensor_sum = np.sum(decrypted_gradients, axis=0)
        buffer = tensor_sum.tobytes()
        # print(len(buffer))
        try:
            c.sendall(buffer)
        except BrokenPipeError as e:
            print(f"Error sending data: {e}")
        
        if aggr_s is not None:
            aggr_time = time.time() - aggr_s 
            aggr_times.append(aggr_time)    
        current_round += 1
        if total_rounds == current_round:
            break
        
    c.close()
    
    print(f"Mean aggregation time: {np.mean(aggr_times)} STD: {np.std(aggr_times)}") 
    print(f"Mean symmetric decryption time:  {np.mean(decrypt_times)} STD: {np.std(decrypt_times)}")
    
    return 
            

if __name__ == '__main__':
    main()
