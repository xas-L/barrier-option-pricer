"""
Utility functions for the Barrier Option Pricer project.
This module can contain helper functions for plotting, data validation,
or other reusable components.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_data(sim_counts: list, prices: list, errors: list, option_type_str: str):
    """
    Plots convergence of option price and Monte Carlo error.

    Parameters:
    -----------
    sim_counts : list
        List of simulation counts.
    prices : list
        List of corresponding estimated option prices.
    errors : list
        List of corresponding Monte Carlo errors (SEM of price).
    option_type_str : str
        String describing the option type for titles.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Option Price
    color = 'tab:blue'
    ax1.set_xlabel('Number of Simulations (N_sim) - Log Scale')
    ax1.set_ylabel('Estimated Option Price ($)', color=color)
    ax1.plot(sim_counts, prices, marker='o', linestyle='-', color=color, label='Option Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # Create a second y-axis for Monte Carlo Error
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Monte Carlo Error (SEM of Price) - Log Scale', color=color)
    ax2.plot(sim_counts, errors, marker='x', linestyle='--', color=color, label='MC Error')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log') # Error often viewed on log scale

    fig.suptitle(f'Convergence Analysis for {option_type_str.replace("_", " ").title()}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    
    plt.show()

def plot_sensitivity_data(param_values: np.ndarray, option_prices: list, 
                           param_name: str, base_param_val: float, base_opt_price: float,
                           option_type_str: str):
    """
    Plots sensitivity analysis results.

    Parameters:
    -----------
    param_values : np.ndarray
        Array of parameter values used in the sensitivity analysis.
    option_prices : list
        List of corresponding estimated option prices.
    param_name : str
        Name of the parameter that was varied.
    base_param_val : float
        The base value of the parameter.
    base_opt_price : float
        The option price at the base parameter value.
    option_type_str : str
        String describing the option type for titles.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, option_prices, marker='o', linestyle='-', label=f'{option_type_str.replace("_", " ").title()} Price')
    plt.scatter([base_param_val], [base_opt_price], color='red', s=100, zorder=5, label=f'Base Case ({param_name}={base_param_val:.3f})')
    
    plt.title(f'Sensitivity of Option Price to {param_name}')
    plt.xlabel(f'{param_name} Value')
    plt.ylabel('Estimated Option Price ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage (can be run directly for testing utils)
    print("Testing plotting utilities (if run directly)...")
    
    # Example data for convergence plot
    sim_counts_example = [1000, 10000, 100000, 1000000]
    prices_example = [5.5, 5.2, 5.15, 5.14]
    errors_example = [0.5, 0.15, 0.05, 0.015]
    plot_convergence_data(sim_counts_example, prices_example, errors_example, "Example Option")

    # Example data for sensitivity plot
    sigma_values_example = np.linspace(0.1, 0.5, 11)
    option_prices_example = 5 + (sigma_values_example - 0.3)**2 + np.random.randn(11) * 0.1 # Dummy data
    plot_sensitivity_data(sigma_values_example, option_prices_example, 
                           "Volatility (sigma)", 0.3, 5.0, "Example Option")

