import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_mnist import BaseMNIST

class LitMNISTLoRA(BaseMNIST):
    """
    A PyTorch Lightning module for MNIST classification using LoRA (Low-Rank Adaptation) technique.
    This class extends the BaseMNIST class and incorporates LoRA parameters to adapt the model's weights.

    Attributes:
        l1 (nn.Linear): First fully connected layer.
        l2 (nn.Linear): Second fully connected layer.
        l3 (nn.Linear): Third fully connected layer for classification.
        dropout (nn.Dropout): Dropout layer for regularization.
        relu (nn.ReLU): ReLU activation function.
        l1_lora_A, l1_lora_B, l2_lora_A, l2_lora_B, l3_lora_A, l3_lora_B (nn.Parameter): LoRA parameters for each layer.
    """

    def __init__(self, config):
        """
        Initializes the LitMNISTLoRA model with the given configuration.

        Args:
            config (object): Configuration object containing model hyperparameters.
        """
        super().__init__(config)
        channels, width, height = self.dims
        self.l1 = nn.Linear(channels * width * height, config.hidden_size)
        self.l2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.l3 = nn.Linear(config.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        # Store configuration options
        self.lora_train_A = config.lora_train_A
        self.lora_train_B = config.lora_train_B
        self.lora_init_method_A = config.lora_init_method_A
        self.lora_init_method_B = config.lora_init_method_B

        # Define LoRA parameters
        self.l1_lora_A = nn.Parameter(torch.empty(channels * width * height, config.lora_rank))
        self.l1_lora_B = nn.Parameter(torch.empty(config.lora_rank, config.hidden_size))
        self.l2_lora_A = nn.Parameter(torch.empty(config.hidden_size, config.lora_rank))
        self.l2_lora_B = nn.Parameter(torch.empty(config.lora_rank, config.hidden_size))
        self.l3_lora_A = nn.Parameter(torch.empty(config.hidden_size, config.lora_rank))
        self.l3_lora_B = nn.Parameter(torch.empty(config.lora_rank, self.num_classes))

        self._init_lora_weights()
        self._freeze_non_lora_weights()

    def _init_lora_weights(self):
        """
        Initializes the LoRA weights based on the specified initialization methods.
        Supports 'zero', 'gaussian', and 'kaiming' initialization methods.
        """
        for n, p in self.named_parameters():
            if 'lora' in n:
                if n.endswith('A'):
                    method = self.lora_init_method_A
                elif n.endswith('B'):
                    method = self.lora_init_method_B
                else:
                    continue
                if method == 'zero':
                    nn.init.zeros_(p)
                elif method == 'gaussian':
                    nn.init.normal_(p, mean=0.0, std=1.0)
                elif method == 'kaiming':
                    if n.endswith('A'):
                        nn.init.kaiming_uniform_(p, a=math.sqrt(10))
                    elif n.endswith('B'):
                        nn.init.kaiming_uniform_(p, a=math.sqrt(10))
                else:
                    raise ValueError(f"Unknown initialization method '{method}' for '{n}'")

    def _freeze_non_lora_weights(self):
        """
        Freezes the non-LoRA weights of the model based on the configuration.
        Only LoRA parameters are trainable if specified in the configuration.
        """
        for n, p in self.named_parameters():
            if 'lora' in n:
                if n.endswith('A') and not self.lora_train_A:
                    p.requires_grad = False
                elif n.endswith('B') and not self.lora_train_B:
                    p.requires_grad = False
            else:
                p.requires_grad = False

    def lora_linear(self, x, layer, lora_A, lora_B):
        """
        Applies a linear transformation with LoRA adaptation.

        Args:
            x (torch.Tensor): Input tensor.
            layer (nn.Linear): Linear layer to apply.
            lora_A (nn.Parameter): LoRA A parameter.
            lora_B (nn.Parameter): LoRA B parameter.

        Returns:
            torch.Tensor: Output tensor after applying the linear transformation and LoRA adaptation.
        """
        h = layer(x)
        h += (x @ lora_A @ lora_B) * self.config.lora_alpha
        return h

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the model's layers and LoRA adaptation.
        """
        x = torch.flatten(x, 1)
        x = self.lora_linear(x, self.l1, self.l1_lora_A, self.l1_lora_B)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lora_linear(x, self.l2, self.l2_lora_A, self.l2_lora_B)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lora_linear(x, self.l3, self.l3_lora_A, self.l3_lora_B)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the model.

        Returns:
            dict: A dictionary containing the optimizer and scheduler configuration.
        """
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

    def merge_lora_weights(self):
        """
        Merges the LoRA weights into the main model weights.
        This is typically done after training to consolidate the LoRA adaptations.
        """
        with torch.no_grad():
            # Merge for l1
            delta_W = (self.l1_lora_A @ self.l1_lora_B) * self.config.lora_alpha
            self.l1.weight.data += delta_W.T
            # Merge for l2
            delta_W = (self.l2_lora_A @ self.l2_lora_B) * self.config.lora_alpha
            self.l2.weight.data += delta_W.T
            # Merge for l3
            delta_W = (self.l3_lora_A @ self.l3_lora_B) * self.config.lora_alpha
            self.l3.weight.data += delta_W.T

    def reset_lora_parameters(self):
        """
        Resets the LoRA parameters to their initial values and re-freezes non-LoRA weights.
        """
        self._init_lora_weights()
        self._freeze_non_lora_weights()