import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision import transforms
from torchmetrics import Accuracy

class BaseMNIST(L.LightningModule):
    def __init__(self, config):
        """Initialize the base MNIST model.
        
        Parameters
        ----------
        config : Config
            Configuration dataclass containing model hyperparameters
            
        Attributes
        ----------
        class_names : list[int]
            List of class indices to use (default: [0, 1, 2, 3, 4])
        min_class : int
            Minimum class index (used for label adjustment)
        num_classes : int
            Number of classes in the dataset
        dims : tuple
            Input dimensions (channels, height, width)
        transform : Compose
            Image preprocessing pipeline
        val_accuracy : Accuracy
            Validation accuracy metric
        test_accuracy : Accuracy
            Test accuracy metric
        """
        super().__init__()
        self.config = config
        self.class_names = [0, 1, 2, 3, 4]
        self.min_class = min(self.class_names)
        self.num_classes = len(self.class_names)
        self.dims = (1, 28, 28)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def common_step(self, batch, batch_idx):
        """Shared forward pass and loss computation.
        
        Parameters
        ----------
        batch : tuple
            Input batch containing (images, labels)
        batch_idx : int
            Index of current batch
            
        Returns
        -------
        tuple
            (images, adjusted_labels, logits, loss)
        """
        x, y = batch
        if self.min_class != 0:
            y = y - self.min_class
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return x, y, logits, loss

    def training_step(self, batch, batch_idx):
        """Perform a single training step.
        
        Parameters
        ----------
        batch : tuple
            Input batch containing (images, labels)
        batch_idx : int
            Index of current batch
            
        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        _, _, _, loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=self.config.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.
        
        Parameters
        ----------
        batch : tuple
            Input batch containing (images, labels)
        batch_idx : int
            Index of current batch
            
        Notes
        -----
        - Computes validation loss and accuracy
        - Logs metrics to progress bar
        """
        x, y, logits, loss = self.common_step(batch, batch_idx)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Perform a single test step.
        
        Parameters
        ----------
        batch : tuple
            Input batch containing (images, labels)
        batch_idx : int
            Index of current batch
            
        Notes
        -----
        - Computes test loss and accuracy
        - Logs metrics to progress bar
        """
        x, y, logits, loss = self.common_step(batch, batch_idx)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        """Configure model optimizers.
        
        Returns
        -------
        torch.optim.Optimizer
            AdamW optimizer with learning rate from config
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer

    @staticmethod
    def get_indices(dataset, class_names):
        """Get indices of samples belonging to specified classes.
        
        Parameters
        ----------
        dataset : Dataset
            MNIST dataset (may be subset)
        class_names : list[int]
            List of class indices to filter
            
        Returns
        -------
        list[int]
            Indices of samples matching specified classes
        """
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            targets = torch.tensor([dataset.dataset.targets[i] for i in dataset.indices])
        else:
            targets = dataset.targets
        indices = [i for i, t in enumerate(targets) if t in class_names]
        return indices

    def create_dataloader(self, dataset):
        """Create a DataLoader for specified dataset.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to create loader for
            
        Returns
        -------
        DataLoader
            Configured DataLoader with class filtering
        """
        idx = self.get_indices(dataset, self.class_names)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=SubsetRandomSampler(idx),
            num_workers=self.config.num_workers
        )

    def prepare_data(self):
        """Download MNIST dataset if not already present.
        
        Notes
        -----
        - Downloads both training and test sets
        - Called automatically by Lightning
        """
        MNIST(self.config.data_dir, train=True, download=True)
        MNIST(self.config.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Set up datasets for current stage.
        
        Parameters
        ----------
        stage : str, optional
            Current stage ('fit', 'test', or None)
            
        Notes
        -----
        - Creates train/val split for 'fit' stage
        - Loads test set for 'test' stage
        """
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.config.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.config.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        """Get training DataLoader.
        
        Returns
        -------
        DataLoader
            Configured DataLoader for training set
        """
        return self.create_dataloader(self.mnist_train)

    def val_dataloader(self):
        """Get validation DataLoader.
        
        Returns
        -------
        DataLoader
            Configured DataLoader for validation set
        """
        return self.create_dataloader(self.mnist_val)

    def test_dataloader(self):
        """Get test DataLoader.
        
        Returns
        -------
        DataLoader
            Configured DataLoader for test set
        """
        return self.create_dataloader(self.mnist_test)
