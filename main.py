"""Entry point to the application."""

import logging
import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from trainers import graphitee

from data.dataset import load_data


# logging.basicConfig(level = logging.INFO)

@hydra.main(config_path="config", config_name="news_recommendation", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the specified application"""

    print(OmegaConf.to_yaml(cfg))
    hydra_output_dir = HydraConfig.get().runtime.output_dir

    if cfg.app == "load_data":
        # downloads the dataset if not done already and returns the processed data.
        # useful for debugging purposes.
        graph, dataset = load_data(cfg=cfg)
        return
    elif cfg.app == "partition_data":
        """ TODO """
    elif cfg.app == "train":
        trainer = graphitee
        if cfg.backend == "gloo":
            n_devices = torch.cuda.device_count()
            devices = [f"{i}" for i in range(n_devices)]

            if "CUDA_VISIBLE_DEVICES" in os.environ:
                devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                n_devices = len(devices)

            torch.multiprocessing.set_start_method('spawn')
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
            p = mp.Process(target=trainer.init_process,
                           args=(0, cfg, hydra_output_dir))
            p.start()
            p.join()
        else:
            raise ValueError(
                f"Backend {cfg.backend} is not supported."
            )
    else:
        raise ValueError(
            f"Backend {cfg.app} is not supported."
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
