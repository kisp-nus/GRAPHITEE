# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: MIT-0
# // see https://github.com/aws-samples/aws-nitro-enclaves-workshop

# // Code here is a simple reference and requires a running enclave and a complete image.

import boto3
import json
import base64
import socket
import subprocess
import numpy as np

KMS_PROXY_PORT="8000"

def get_plaintext(credentials):
    """
    prepare inputs and invoke decrypt function
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


    s = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    cid = socket.VMADDR_CID_ANY
    port = 5000
    s.bind((cid, port))
    s.listen()
    print(f"Started server on port {port} and cid {cid}")

    while True:
        c, addr = s.accept()
        payload = c.recv(4096)
        r = {}
        credentials = json.loads(payload.decode())
        
        print("Received the tensors")
        tensors = get_plaintext(credentials)
        
        if isinstance(tensors[0], str) and tensors[0].startswith("KMS Error. Decryption Failed."):
            r["error"] = tensors[0]
        else:

            tensor_sum = np.sum(tensors)

            r["tensor_sum"] = tensor_sum.item()
            print(tensor_sum.item())
            print(json.dumps(r))
            print(str.encode(json.dumps(r)))
            
            
        try:
            c.send(str.encode(json.dumps(r)))
        except BrokenPipeError as e:
            print(f"Error sending data: {e}")
        finally:
            c.close()

        
        # plaintext = get_plaintext(credentials)
        # print(plaintext)

        # if plaintext.startswith("KMS Error. Decryption Failed."):
        #     r["error"] = plaintext
        # else:
        #     last_four = plaintext[-4:]
        #     r["last_four"] = last_four

        # c.send(str.encode(json.dumps(r)))

        # c.close()

if __name__ == '__main__':
    main()
