"""Implement dataset related functions"""

import logging
from data.mind_dataset import MIND_DGL
import dgl  # type: ignore
from typing import Tuple

logger = logging.getLogger(__name__)


# Potential Datasets to add:
# - Addressa; MovieLens; AmazonBooks; Reddit; Yelp; etc.

def load_data(cfg=None) -> Tuple[dgl.DGLGraph, int, int]:
    """Load dataset

    Parameters
    ----------
    dataset_name : str
        Dataset name

    Returns
    -------
    dgl.DGLGraph
        Graph in DGL format
    dgl.data.DGLDataset
        Dataset in DGL format
    """
    if cfg.dataset_name == "mind":
        dataset = MIND_DGL(cfg, force_reload=cfg.force_reload)
        graph = dataset.graph
        return graph, dataset
    else:
        raise ValueError(f"Dataset {cfg.dataset_name} is not supported")
