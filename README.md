# Monte Carlo Pricer for Barrier Options

A Python-based Monte Carlo simulation engine for pricing various types of barrier options. The simulation of the underlying asset price paths is based on Geometric Brownian Motion (GBM), with user-defined parameters. The program calculates the option price and provides a confidence interval for the estimated price.

This project is designed to demonstrate knowledge of exotic derivatives (specifically path-dependent barrier options) and a widely used pricing technique (Monte Carlo simulation) for such instruments.

## Features

* **Comprehensive Barrier Option Support:**
    * Down-and-Out (Call/Put)
    * Up-and-Out (Call/Put)
    * Down-and-In (Call/Put)
    * Up-and-In (Call/Put)
* **Geometric Brownian Motion (GBM):** Simulates asset price paths under the risk-neutral measure.
* **Flexible Monitoring:**
    * **Discrete Monitoring:** Barrier checks at specified time steps.
    * **Continuous Monitoring Approximation:** Implements a continuity correction (Broadie, Glasserman, Kou, 1997) to adjust the barrier for discrete simulations aiming to mimic continuous monitoring.
* **Variance Reduction:** Option to use antithetic variates to potentially improve simulation efficiency.
* **Statistical Output:** Provides the estimated option price along with a user-defined confidence interval (e.g., 95%).
* **Analysis Tools:**
    * **Convergence Analysis:** Shows how the option price and Monte Carlo error converge as the number of simulations increases.
    * **Sensitivity Analysis:** Analyzes the impact of changes in key input parameters (S0, K, B, T, r, sigma) on the option price.
* **User-Friendly CLI:** Interactive command-line interface to input parameters and run the pricer.
* **Educational Notebooks:** Jupyter notebooks demonstrating GBM simulation, option pricing examples, convergence, sensitivity, and the impact of continuity correction.

## Theoretical Background

### Barrier Options
Barrier options are path-dependent exotic options whose payoff depends on whether the underlying asset's price reaches a predetermined barrier level ($B$) during the option's life.
-   **Knock-Out (Out) Options:** Expire worthless if the barrier is breached.
-   **Knock-In (In) Options:** Become active (i.e., turn into a standard European option) only if the barrier is breached.
The direction of the barrier (Up or Down) relative to the initial asset price also defines the option type. (Reference: "An Introduction to Exotic Option Pricing" by Peter Buchen, Chapter 7)

### Geometric Brownian Motion (GBM)
The price of the underlying asset ($S_t$) is assumed to follow GBM under the risk-neutral measure:
$dS_t = r S_t dt + \sigma S_t dW_t$
For discrete simulation steps ($\Delta t$), this is often implemented as:
$S_{t+\Delta t} = S_t \exp((r - \frac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} Z)$
where $Z \sim N(0,1)$.

### Monte Carlo Simulation for Option Pricing
The price of an option can be estimated as the expected discounted payoff under the risk-neutral measure:
$Price = e^{-rT} \mathbb{E_Q}[\text{Payoff}(S_T)]$
Monte Carlo simulation estimates this expectation by:
1.  Simulating a large number ($N_{sim}$) of possible price paths for the underlying asset until maturity $T$.
2.  Calculating the option's payoff for each path, considering any barrier conditions.
3.  Averaging these payoffs.
4.  Discounting the average payoff back to the present value using the risk-free rate $r$.

## Repository Structure

monte-carlo-barrier-option-pricer/│├── .gitignore├── LICENSE├── README.md│├── barrier_option_pricer/   # Main application package│   ├── init.py│   ├── pricer.py            # Core BarrierOptionsPricer class and logic│   ├── utils.py             # Optional: Helper functions (e.g., for plotting)│   └── cli.py               # Optional: CLI logic if separated│├── notebooks/                 # Jupyter notebooks for demos and analysis│   ├── 01_GBM_Simulation_Demo.ipynb│   ├── 02_Barrier_Option_Pricing_Examples.ipynb│   ├── 03_Convergence_Analysis.ipynb│   ├── 04_Sensitivity_Analysis.ipynb│   └── 05_Continuity_Correction_Impact.ipynb│├── requirements.txt         # Python package dependencies│├── setup.py                 # For making the project an installable package│└── tests/                     # Unit and integration tests├── init.py└── ... (test files for different modules)
## Prerequisites

* Python (>=3.8 recommended)
* The libraries listed in `requirements.txt`:
    * `numpy`
    * `scipy`
    * `matplotlib`

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/monte-carlo-barrier-option-pricer.git](https://github.com/yourusername/monte-carlo-barrier-option-pricer.git) # Replace with your repo URL
    cd monte-carlo-barrier-option-pricer
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Install as a package:**
    If you want to install the pricer as a package (e.g., for easier imports in other projects or if you define entry points):
    ```bash
    pip install .
    # For development mode (changes are reflected immediately)
    # pip install -e .
    ```

## Usage

The primary way to use the pricer is via its command-line interface, which is part of the `pricer.py` script (or `cli.py` if you separate it).

**Running the Pricer:**

Navigate to the `barrier_option_pricer` directory (or the root directory if `setup.py` is used to create an entry point).

If `pricer.py` contains the `main()` function:
```bash
python pricer.py
or if run as a module from the root directory:python -m barrier_option_pricer.pricer
The script will prompt you to either use default parameters or enter custom parameters for:Initial asset price (S0​)Strike price (K)Barrier level (B)Time to maturity (T in years)Risk-free rate (r)Volatility (σ)Option type (e.g., down_and_out_call, up_and_in_put)Number of simulations (Nsim​)Number of time steps (Nsteps​)Monitoring type (discrete or continuous_approx)(Optional) Whether to perform additional analysis (Convergence, Sensitivity).Example Parameters (Defaults in the script):S0: 100.0K: 100.0B: 90.0T: 1.0 yearr: 0.05 (5%)sigma: 0.20 (20%)option_type: down_and_out_callN_sim: 100,000N_steps: 252 (daily)monitoring_type: discreteOutput ExplanationThe pricer will output:Option Details: The parameters used for the pricing.Simulation Parameters: Number of simulations, time steps, monitoring type.Pricing Results:Estimated Option Price.Confidence Interval (e.g., 95%) for the price.Monte Carlo Error (Standard Error of the Mean Price).Additional Statistics:Mean undiscounted payoff.Standard deviation of payoffs.Percentage of paths that hit the barrier (approximate).Computation time.If additional analyses (Convergence or Sensitivity) are run, their respective results and plots will also be displayed/generated.Jupyter NotebooksThe notebooks/ directory contains several Jupyter notebooks to demonstrate and explore different aspects of the project:

- 01_GBM_Simulation_Demo.ipynb: Illustrates the simulation of asset paths using Geometric Brownian Motion and visualizes the paths and final price distribution.
- 02_Barrier_Option_Pricing_Examples.ipynb: Shows how to use the BarrierOptionsPricer class to price different types of barrier options, including examples with continuous monitoring approximation and antithetic variates.
- 03_Convergence_Analysis.ipynb: Demonstrates how the Monte Carlo estimate and its error converge as the number of simulations increases.
- 04_Sensitivity_Analysis.ipynb: Shows how the option price changes in response to variations in key input parameters.
- 05_Continuity_Correction_Impact.ipynb: Focuses on the difference in pricing when using discrete monitoring versus the continuity correction for approximating continuous monitoring.

To run these notebooks, ensure you have Jupyter Notebook or JupyterLab installed (pip install notebook jupyterlab).TestingThe tests/ directory is intended for unit and integration tests to ensure the correctness and robustness of the pricer's components. (Note: Test scripts would need to be developed using a framework like pytest or unittest).LimitationsModel Assumptions: The pricer relies on the assumptions of the Black-Scholes model (e.g., GBM, constant volatility, constant risk-free rate, no dividends unless explicitly modeled in r). Real-world asset prices may not always follow these assumptions. Computational Cost: Monte Carlo simulations can be computationally intensive, especially for high accuracy (large Nsim​) or many time steps (Nsteps​).Continuity Correction: The continuity correction is an approximation for continuous monitoring and may not be perfectly accurate, especially with few time steps.Early Exercise: This pricer is for European-style barrier options (payoff at maturity). It does not handle American-style barrier options with early exercise features. Potential Future Enhancements: Implement additional variance reduction techniques (e.g., control variates). Adding support for options on assets paying discrete or continuous dividends. Sophisticated GUI. Extend to other exotic options or stochastic processes (e.g., jump-diffusion models, stochastic volatility). Implement analytical pricers for certain barrier options for comparison.