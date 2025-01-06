import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from callbacks import LoRAMergeCallback
from config import Config
from models.lit_mnist import LitMNIST
from models.lit_mnist_lora import LitMNISTLoRA

def train_model(model, config, is_lora_train):
    """Train a MNIST model with optional LoRA training configuration.
    
    Parameters
    ----------
    model : LightningModule
        The model to train (either baseline or LoRA variant)
    config : Config
        Configuration dataclass containing training parameters
    is_lora_train : bool
        Whether this is a LoRA training run (affects callback setup)
        
    Returns
    -------
    Trainer
        The trained PyTorch Lightning trainer instance
        
    Notes
    -----
    - Sets up EarlyStopping callback by default
    - Adds LoRAMergeCallback if merge_frequency > 0 and is_lora_train
    - Uses CSVLogger to save training logs
    """
    callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    if config.merge_frequency > 0 and is_lora_train:
        callbacks.append(LoRAMergeCallback(config.merge_frequency))

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=config.max_epochs,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=callbacks,    deterministic=True
    )
    trainer.fit(model)
    trainer.test()
    return trainer

def run_baseline_experiment():
    """Run baseline MNIST model training experiment.
    
    Returns
    -------
    float
        Test accuracy of the baseline model
        
    Notes
    -----
    - Checks for existing model checkpoint to avoid retraining
    - If no checkpoint exists, trains new model and saves checkpoint
    - Uses default Config parameters for baseline training
    """
    model_filepath = "base_model.ckpt"
    if os.path.isfile(model_filepath):
        print("Found existing base model checkpoint. Skipping training of the base model")
        trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            logger=CSVLogger(save_dir="logs/"),
        )
        model = LitMNIST.load_from_checkpoint(model_filepath, config=Config())
    else:
        config = Config()
        model = LitMNIST(config)
        trainer = train_model(model, config, False)
        trainer.save_checkpoint("base_model.ckpt")
        torch.save(model.state_dict(), 'base_model.pt')
    return trainer.test(model)[0]['test_acc']

def run_lora_experiment(rank, train_A=True, train_B=True, init_method_A='kaiming', init_method_B='zero', merge_frequency=1):
    """Run LoRA training experiment with specified configuration.
    
    Parameters
    ----------
    rank : int
        LoRA rank (dimension of low-rank matrices)
    train_A : bool, optional
        Whether to train LoRA A matrix (default: True)
    train_B : bool, optional
        Whether to train LoRA B matrix (default: True)
    init_method_A : str, optional
        Initialization method for A matrix (default: 'kaiming')
    init_method_B : str, optional
        Initialization method for B matrix (default: 'zero')
    merge_frequency : int, optional
        Frequency of LoRA weight merging (default: 1)
        
    Returns
    -------
    float
        Test accuracy of the LoRA model
        
    Notes
    -----
    - Loads weights from pre-trained baseline model
    - Uses modified Config with LoRA-specific parameters
    - Trains model with LoRA-specific callbacks
    """
    config = Config(
        lora_rank=rank,
        lora_train_A=train_A,
        lora_train_B=train_B,
        lora_init_method_A=init_method_A,
        lora_init_method_B=init_method_B,
        merge_frequency=merge_frequency
    )
    state_dict = torch.load("base_model.pt")
    model = LitMNISTLoRA(config)
    model.load_state_dict(state_dict, strict=False)
    model.class_names = [5, 6, 7, 8, 9]
    model.min_class = min(model.class_names)
    trainer = train_model(model, config, True)
    return trainer.test(model)[0]['test_acc']
