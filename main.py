"""Entry point to the application."""

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

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



    train = trainers.pos_neg_hetero_trainer_networking

    torch.multiprocessing.set_start_method('spawn')
    master_p = mp.Process(target=train.init_master, args=(cfg, hydra_output_dir))
    master_p.start()
    master_p.join()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
