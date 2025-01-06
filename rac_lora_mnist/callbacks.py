from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

class LoRAMergeCallback(Callback):
    def __init__(self, merge_frequency):
        """Initialize the LoRA merge callback.
        
        Parameters
        ----------
        merge_frequency : int
            Frequency (in epochs) at which to merge LoRA weights
        """
        self.merge_frequency = merge_frequency

    def on_train_epoch_end(self, trainer, pl_module):
        """Callback executed at the end of each training epoch.
        
        Merges LoRA weights and resets parameters according to merge frequency.
        
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning trainer instance
        pl_module : pytorch_lightning.LightningModule
            The Lightning module containing LoRA layers
            
        Notes
        -----
        This callback will:
        1. Check if current epoch matches merge frequency
        2. Merge LoRA weights into base model weights
        3. Reset LoRA parameters for next training phase
        """
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.merge_frequency == 0:
            print("Done merging!")
            pl_module.merge_lora_weights()
            pl_module.reset_lora_parameters()
