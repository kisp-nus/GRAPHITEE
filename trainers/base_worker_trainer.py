"""Implement the base classes that are run by a worker"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
import dgl # type: ignore
import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.loader import NeighborLoader # type: ignore

class BaseWorkerTrainer(ABC):
    """Implement training process for a worker

    Parameters
    ----------
    """

    @abstractmethod
    def train_model(
        self,
        model: nn.Module,
        traindata: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor],
        optimizer: Optimizer,
        loss_function: Callable,
        local_epochs: int = 1,
    ) -> None:
        """Train the model

        Parameters
        ----------
        model : nn.Module
            Model to be trained
        traindata : Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]
            DGLGraph, input features and labels for train
        optimizer : Optimizer
            Optimizer for training
        loss_function : Callable
            Loss function
        local_epochs : int, optional
            Number of local epochs, by default 1
        """

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        data: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor],
        metric_fn: List[Callable],
        loss_function: Callable,
    ) -> Dict[str, float]:
        """Evaluate the model"""
