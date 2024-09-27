"""Entry point to the application."""

import logging
import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from data.dataset import load_data, create_partitions
from topics import compute_topics, eval_user_topics
import trainers.pos_neg_hetero_trainer_networking

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# logging.basicConfig(level = logging.INFO)

@hydra.main(config_path="conf", config_name="news_recommendation", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the specified application"""

    print(OmegaConf.to_yaml(cfg))
    # get the hydra output directory
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    
    # # temporary code to handle keys locally for test
    # private_key = rsa.generate_private_key(
    #     public_exponent=65537,
    #     key_size=2048,
    #     backend=default_backend()
    # )
    # public_key = private_key.public_key()
    # with open("private_key.pem", "wb") as private_file:
    #     private_file.write(
    #         private_key.private_bytes(
    #             encoding=serialization.Encoding.PEM,
    #             format=serialization.PrivateFormat.PKCS8,
    #             encryption_algorithm=serialization.NoEncryption()
    #         )
    #     )
    # with open("public_key.pem", "wb") as public_file:
    #     public_file.write(
    #         public_key.public_bytes(
    #             encoding=serialization.Encoding.PEM,
    #             format=serialization.PublicFormat.SubjectPublicKeyInfo
    #         )
    #     )
    if cfg.app == "load":
        cfg.dataset_dir = "./datasets/partitions/0"
        graph,  dataset = load_data(**cfg.dataset.download, cfg=cfg)
        return
    elif cfg.app == "compute_topics":
        topics_df = compute_topics(cfg)
    elif cfg.app == "eval_user_topics":
        results = eval_user_topics(cfg)
    elif cfg.app == "create_partitions":
        create_partitions(cfg)
    elif cfg.app == "hetero_pos_neg_train":
        if cfg.federated:
            train = trainers.pos_neg_hetero_trainer_networking
            if cfg.distributed.backend == "gloo":
                n_devices = torch.cuda.device_count()
                devices = [f"{i}" for i in range(n_devices)]

                if n_devices > 0:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                        n_devices = len(devices)
                    os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
                    
                torch.multiprocessing.set_start_method('spawn')
                # master_p = mp.Process(target=train.init_master, args=(cfg, hydra_output_dir))

                worker_p = mp.Process(target=train.init_process, args=(cfg.rank, cfg.num_partitions + 1, cfg, hydra_output_dir))
                
                # master_p.start()
                worker_p.start()
                # master_p.join()
                worker_p.join()
        else:
            train = trainers.pos_neg_hetero_trainer
            if cfg.distributed.backend == "gloo":
                n_devices = torch.cuda.device_count()
                devices = [f"{i}" for i in range(n_devices)]
                
                if n_devices > 0:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                        n_devices = len(devices)
                        
                    torch.multiprocessing.set_start_method('spawn')
                    os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
                p = mp.Process(target=train.init_process, args=(0, cfg.num_partitions, cfg, hydra_output_dir))
                p.start()
                p.join()              
    else:
        raise ValueError(
            f"Unknown application {cfg.app} is not supported."
        )

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
