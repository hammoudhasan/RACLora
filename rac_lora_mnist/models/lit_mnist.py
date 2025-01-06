import torch
from torch import nn
from torch.nn import functional as F

from models.base_mnist import BaseMNIST

class LitMNIST(BaseMNIST):
    def __init__(self, config):
        """Initialize the standard MNIST model.
        
        Parameters
        ----------
        config : Config
            Configuration dataclass containing model hyperparameters
            
        Notes
        -----
        - Inherits from BaseMNIST for common functionality
        - Defines fully connected layers with ReLU activations
        - Uses dropout for regularization
        """
        super().__init__(config)
        channels, width, height = self.dims
        self.l1 = nn.Linear(channels * width * height, config.hidden_size)
        self.l2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.l3 = nn.Linear(config.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns
        -------
        torch.Tensor
            Log probabilities for each class (log_softmax output)
            
        Notes
        -----
        - Flattens input before passing through fully connected layers
        - Applies ReLU activations and dropout between layers
        """
        x = torch.flatten(x, 1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l3(x)
        return F.log_softmax(x, dim=1)
