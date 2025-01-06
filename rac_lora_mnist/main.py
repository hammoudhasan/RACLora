import torch
from train import run_baseline_experiment, run_lora_experiment
from utils import plot_results

if __name__ == "__main__":
    """
    Main script to run baseline and LoRA experiments for MNIST classification.

    This script performs the following steps:
    1. Runs a baseline experiment which is regular full training on five classes on MNIST.
    2. Runs LoRA experiments for different ranks (1, 2, 4, 8) to evaluate the impact of LoRA on model performance for learning the remaining five classes.
    3. Prints the test accuracy for each LoRA rank.
    4. Plots the results comparing the baseline and LoRA experiments.

    The results are stored in a dictionary and visualized using the `plot_results` function.
    """
    # Run baseline experiment to establish a performance benchmark
    baseline_acc = run_baseline_experiment()

    # Dictionary to store results for different LoRA ranks
    results = {}

    # Run LoRA experiments for different ranks
    for rank in [1, 2, 4, 8]:
        results[rank] = run_lora_experiment(
            rank,
            train_A=True,  # Whether to train the LoRA A matrix
            train_B=True,  # Whether to train the LoRA B matrix
            init_method_A='zero',  # Initialization method for LoRA A matrix
            init_method_B='gaussian',  # Initialization method for LoRA B matrix
            merge_frequency=2  # Frequency to merge LoRA weights into the model
        )
        # Print test accuracy for the current rank
        print(f"Rank {rank}, Test Accuracy: {results[rank]}")

    # Plot the results comparing baseline and LoRA experiments
    plot_results(results, baseline_acc=baseline_acc)