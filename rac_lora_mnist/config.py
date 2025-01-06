from dataclasses import dataclass
from lightning import seed_everything

# Seed everything for reproducibility
seed_everything(42, workers=True)

@dataclass
class Config:
    """
    A configuration class to store hyperparameters and settings for the MNIST LoRA model.

    Attributes:
        data_dir (str): Directory where the dataset is stored. Default is './data'.
        hidden_size (int): Size of the hidden layers in the model. Default is 64.
        lr (float): Learning rate for the optimizer. Default is 2e-4.
        batch_size (int): Batch size for training and evaluation. Default is 128.
        num_workers (int): Number of workers for data loading. Default is 16.
        max_epochs (int): Maximum number of training epochs. Default is 50.
        lora_rank (int): Rank of the LoRA matrices. Default is 1.
        lora_alpha (float): Scaling factor for LoRA weights. Default is 1.
        lora_train_A (bool): Whether to train the LoRA A matrix. Default is True.
        lora_train_B (bool): Whether to train the LoRA B matrix. Default is True.
        lora_init_method_A (str): Initialization method for LoRA A matrix. 
            Options: 'zero', 'gaussian', 'kaiming'. Default is 'kaiming'.
        lora_init_method_B (str): Initialization method for LoRA B matrix. 
            Options: 'zero', 'gaussian', 'kaiming'. Default is 'zero'.
        merge_frequency (int): Frequency (in epochs) to merge LoRA weights into the main model weights. 
            Default is 1.
    """
    data_dir: str = './data'
    hidden_size: int = 64
    lr: float = 2e-4
    batch_size: int = 128
    num_workers: int = 16
    max_epochs: int = 50
    lora_rank: int = 1
    lora_alpha: float = 1
    lora_train_A: bool = True  # Whether to train the A matrix
    lora_train_B: bool = True  # Whether to train the B matrix
    lora_init_method_A: str = 'kaiming'  # 'zero', 'gaussian', 'kaiming'
    lora_init_method_B: str = 'zero'     # 'zero', 'gaussian', 'kaiming'
    merge_frequency: int = 1  # Frequency to merge LoRA weights into the model