# RACLora
<p align="center">
    <a href="https://arxiv.org/abs/2410.08305">
        <img src="https://img.shields.io/badge/arXiv-2410.08305-b31b1b.svg" alt="arXiv">
    </a>
</p>

<p align="center">
    <p align="center">
        <strong style="font-size: 1.2em;">Grigory Malinovsky, Umberto Michieli, Hasan Abed Al Kader Hammoud, Taha Ceritli, Hayder Elesedy, Mete Ozay, Peter Richtárik</strong>
    </p>

> Work was completed during Grigory's internship at Samsung Research UK (SRUK).

## Abstract

Fine-tuning has become a popular approach to adapting large foundational models to specific tasks. As the size of models and datasets grows, parameter-efficient fine-tuning techniques are increasingly important. One of the most widely used methods is Low-Rank Adaptation (LoRA), with adaptation update expressed as the product of two low-rank matrices. While LoRA was shown to possess strong performance in fine-tuning, it often under-performs when compared to full-parameter fine-tuning (FPFT). Although many variants of LoRA have been extensively studied empirically, their theoretical optimization analysis is heavily under-explored. The starting point of our work is a demonstration that LoRA and its two extensions, Asymmetric LoRA and Chain of LoRA, indeed encounter convergence issues. To address these issues, we propose Randomized Asymmetric Chain of LoRA (RAC-LoRA) -- a general optimization framework that rigorously analyzes the convergence rates of LoRA-based methods. Our approach inherits the empirical benefits of LoRA-style heuristics, but introduces several small but important algorithmic modifications which turn it into a provably convergent method. Our framework serves as a bridge between FPFT and low-rank adaptation. We provide provable guarantees of convergence to the same solution as FPFT, along with the rate of convergence. Additionally, we present a convergence analysis for smooth, non-convex loss functions, covering gradient descent, stochastic gradient descent, and federated learning settings. Our theoretical findings are supported by experimental results.

## Project Description

This project implements baseline and LoRA (Low-Rank Adaptation) experiments for MNIST classification. 
The main script `main.py` performs the following steps:
1. Runs a baseline experiment with regular full training on five classes of the MNIST dataset.
2. Runs LoRA experiments for different ranks (1, 2, 4, 8) to evaluate the impact of LoRA on model performance for learning the remaining five classes of MNIST.

> This setting is inspired by https://github.com/sunildkumar/lora_from_scratch/

## Key Configuration Parameters

The following are key configuration parameters defined in `config.py`:

- `data_dir`: Directory where the dataset is stored (default: './data').
- `hidden_size`: Size of the hidden layers in the model (default: 64).
- `lr`: Learning rate for the optimizer (default: 2e-4).
- `batch_size`: Batch size for training and evaluation (default: 128).
- `num_workers`: Number of workers for data loading (default: 16).
- `max_epochs`: Maximum number of training epochs (default: 50).
- `lora_rank`: Rank of the LoRA matrices (default: 1).
- `lora_alpha`: Scaling factor for LoRA weights (default: 1).
- `lora_train_A`: Whether to train the LoRA A matrix (default: True).
- `lora_train_B`: Whether to train the LoRA B matrix (default: True).
- `lora_init_method_A`: Initialization method for LoRA A matrix (default: 'kaiming').
- `lora_init_method_B`: Initialization method for LoRA B matrix (default: 'zero').
- `merge_frequency`: Frequency (in epochs) to merge LoRA weights into the main model weights (default: 1).

> Those settings can be used to activate LoRa, Chain of LoRa, Asymmetric LoRa, and RACLoRa (ours).

## Citation

```bibtex
@article{malinovsky2024randomized,
    title={Randomized Asymmetric Chain of LoRA: The First Meaningful Theoretical Framework for Low-Rank Adaptation},
    author={Malinovsky, Grigory and Michieli, Umberto and Hammoud, Hasan Abed Al Kader and Ceritli, Taha and Elesedy, Hayder and Ozay, Mete and Richt{\'a}rik, Peter},
    journal={arXiv preprint arXiv:2410.08305},
    year={2024}
}
```
