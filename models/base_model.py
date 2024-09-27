"""Implement the abstract base class for GNN model"""

from abc import ABC, abstractmethod
import torch
from torch import nn
from torch_geometric.nn import MessagePassing # type: ignore



class BaseGNN(nn.Module, ABC):
    """Abstract base class for GNN"""

    @abstractmethod
    def get_nth_layer(self, n: int) -> nn.Module:
        """Get the nth layer of the model

        Parameters
        ----------
        n : int
            Index of the layer

        Returns
        -------
        nn.Module
            The nth layer, object with forward method
        """

class BaseCustomConv(MessagePassing, ABC):
    """Abstract base class for custom convolution"""

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        Any

        Returns
        -------
        torch.Tensor
            Output feature matrix
        """
