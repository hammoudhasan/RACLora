import matplotlib.pyplot as plt


def plot_results(results, baseline_acc):
    """Plot LoRA rank vs test accuracy results.
    
    Generates a plot comparing LoRA model accuracy at different ranks against
    a baseline model accuracy. Saves the plot as 'rank_vs_accuracy.png'.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping LoRA ranks to test accuracies
        Example: {1: 0.85, 2: 0.87, 4: 0.89}
    baseline_acc : float
        Baseline model accuracy to compare against
        
    Returns
    -------
    None
        Displays and saves the plot but returns nothing
        
    Examples
    --------
    >>> results = {1: 0.85, 2: 0.87, 4: 0.89}
    >>> plot_results(results, baseline_acc=0.82)
    """
    fig, ax = plt.subplots()
    ranks = list(results.keys())
    accuracies = list(results.values())

    # Scatter plot for LoRA results
    ax.plot(ranks, accuracies, marker='o', linestyle='-', color='blue', label='LoRA')

    # Baseline accuracy as a horizontal line
    ax.axhline(y=baseline_acc, color='orange', linestyle='--', label='Baseline')

    # Labeling the plot
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('LoRA Rank vs. Test Accuracy')
    ax.legend()

    # Improve readability
    plt.xticks(ranks)
    plt.grid(True)

    # Save and show the plot
    plt.savefig('rank_vs_accuracy.png')
    plt.show()
