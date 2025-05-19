import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
import time

class BarrierOptionsPricer:
    """
    Monte Carlo pricer for barrier options using Geometric Brownian Motion.
    
    Supports all four types of barrier options (Down-and-Out, Up-and-Out, 
    Down-and-In, Up-and-In) for both calls and puts.
    """
    
    def __init__(self):
        """Initialize the pricer."""
        self.valid_option_types = [
            'down_and_out_call', 'down_and_out_put',
            'up_and_out_call', 'up_and_out_put',
            'down_and_in_call', 'down_and_in_put',
            'up_and_in_call', 'up_and_in_put'
        ]
    
    def simulate_gbm_paths(self, S0: float, r: float, sigma: float, T: float, 
                          N_sim: int, N_steps: int) -> np.ndarray:
        """
        Simulate asset price paths using Geometric Brownian Motion.
        
        Parameters:
        -----------
        S0 : float
            Initial asset price
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility (annualized)
        T : float
            Time to maturity (in years)
        N_sim : int
            Number of Monte Carlo paths
        N_steps : int
            Number of time steps for discretizing the path
        
        Returns:
        --------
        np.ndarray
            Array of shape (N_sim, N_steps + 1) containing all price paths
        """
        # Calculate time step
        dt = T / N_steps
        
        # Pre-calculate drift and diffusion terms
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random numbers for all paths and time steps
        random_shocks = np.random.normal(0, 1, (N_sim, N_steps))
        
        # Initialize paths array
        paths = np.zeros((N_sim, N_steps + 1))
        paths[:, 0] = S0  # Set initial price
        
        # Generate paths efficiently using vectorized operations
        for i in range(N_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion * random_shocks[:, i])
        
        return paths
    
    def apply_continuity_correction(self, B: float, sigma: float, T: float, 
                                  N_steps: int, option_type: str) -> float:
        """
        Apply continuity correction for discrete monitoring to approximate continuous barrier.
        
        Parameters:
        -----------
        B : float
            Original barrier level
        sigma : float
            Volatility
        T : float
            Time to maturity
        N_steps : int
            Number of time steps
        option_type : str
            Type of barrier option
        
        Returns:
        --------
        float
            Adjusted barrier level
        """
        dt = T / N_steps
        correction_factor = 0.5826 * sigma * np.sqrt(dt)
        
        # Determine sign of correction based on option type
        if 'down' in option_type:
            if 'out' in option_type:
                # Down-and-out: lower the barrier
                B_adj = B * np.exp(-correction_factor)
            else:
                # Down-and-in: raise the barrier
                B_adj = B * np.exp(correction_factor)
        else:  # 'up' in option_type
            if 'out' in option_type:
                # Up-and-out: raise the barrier
                B_adj = B * np.exp(correction_factor)
            else:
                # Up-and-in: lower the barrier
                B_adj = B * np.exp(-correction_factor)
        
        return B_adj
    
    def calculate_barrier_payoff(self, path: np.ndarray, K: float, B: float, 
                               option_type: str, T: float, r: float,
                               monitoring_type: str = 'discrete') -> float:
        """
        Calculate the payoff for a single path given the barrier conditions.
        
        Parameters:
        -----------
        path : np.ndarray
            Single price path
        K : float
            Strike price
        B : float
            Barrier level
        option_type : str
            Type of barrier option
        T : float
            Time to maturity
        r : float
            Risk-free rate
        monitoring_type : str
            'discrete' or 'continuous_approx'
        
        Returns:
        --------
        float
            Payoff for this path
        """
        if option_type not in self.valid_option_types:
            raise ValueError(f"Invalid option type. Must be one of {self.valid_option_types}")
        
        # Apply continuity correction if requested
        if monitoring_type == 'continuous_approx':
            sigma_est = 0.2  # Default volatility estimate for correction
            N_steps = len(path) - 1
            B_adj = self.apply_continuity_correction(B, sigma_est, T, N_steps, option_type)
        else:
            B_adj = B
        
        # Initialize barrier conditions
        knocked_out = False
        knocked_in = False
        
        # Monitor barrier throughout the path
        for price in path:
            if 'down_and_out' in option_type:
                if price <= B_adj:
                    knocked_out = True
                    break
            elif 'up_and_out' in option_type:
                if price >= B_adj:
                    knocked_out = True
                    break
            elif 'down_and_in' in option_type:
                if price <= B_adj:
                    knocked_in = True
                    break
            elif 'up_and_in' in option_type:
                if price >= B_adj:
                    knocked_in = True
                    break
        
        # Calculate intrinsic payoff at maturity
        S_T = path[-1]
        
        if 'call' in option_type:
            intrinsic_payoff = max(S_T - K, 0)
        else:  # put
            intrinsic_payoff = max(K - S_T, 0)
        
        # Apply barrier conditions to determine final payoff
        payoff = 0.0
        
        if 'out' in option_type:
            # For knock-out options, pay off only if barrier was never hit
            if not knocked_out:
                payoff = intrinsic_payoff
        else:  # 'in' in option_type
            # For knock-in options, pay off only if barrier was hit
            if knocked_in:
                payoff = intrinsic_payoff
        
        return payoff
    
    def monte_carlo_pricer(self, S0: float, K: float, B: float, T: float,
                          r: float, sigma: float, option_type: str,
                          N_sim: int, N_steps: int,
                          monitoring_type: str = 'discrete',
                          confidence_level: float = 0.95,
                          antithetic: bool = False) -> Tuple[float, float, float, dict]:
        """
        Price barrier options using Monte Carlo simulation.
        
        Parameters:
        -----------
        S0 : float
            Initial asset price
        K : float
            Strike price
        B : float
            Barrier level
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility (annualized)
        option_type : str
            Type of barrier option
        N_sim : int
            Number of Monte Carlo paths
        N_steps : int
            Number of time steps
        monitoring_type : str
            'discrete' or 'continuous_approx'
        confidence_level : float
            Confidence level for the interval (default 0.95)
        antithetic : bool
            Whether to use antithetic variates for variance reduction
        
        Returns:
        --------
        Tuple[float, float, float, dict]
            option_price, confidence_interval_lower, confidence_interval_upper, statistics
        """
        start_time = time.time()
        
        # Adjust number of simulations for antithetic variates
        if antithetic:
            effective_sims = N_sim // 2
        else:
            effective_sims = N_sim
        
        # Generate all paths
        paths = self.simulate_gbm_paths(S0, r, sigma, T, effective_sims, N_steps)
        
        # Apply antithetic variates if requested
        if antithetic:
            # Generate antithetic paths by negating the random shocks
            dt = T / N_steps
            drift = (r - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)
            
            # Calculate log returns for original paths
            log_returns = np.diff(np.log(paths), axis=1)
            
            # Create antithetic log returns
            antithetic_log_returns = -log_returns
            
            # Generate antithetic paths
            antithetic_paths = np.zeros_like(paths)
            antithetic_paths[:, 0] = S0
            
            for i in range(N_steps):
                antithetic_paths[:, i + 1] = antithetic_paths[:, i] * np.exp(antithetic_log_returns[:, i])
            
            # Combine original and antithetic paths
            all_paths = np.vstack([paths, antithetic_paths])
        else:
            all_paths = paths
        
        # Calculate payoffs for all paths
        all_payoffs = []
        for path in all_paths:
            payoff = self.calculate_barrier_payoff(path, K, B, option_type, T, r, monitoring_type)
            all_payoffs.append(payoff)
        
        all_payoffs = np.array(all_payoffs)
        
        # Calculate discounted option price
        mean_payoff = np.mean(all_payoffs)
        option_price = mean_payoff * np.exp(-r * T)
        
        # Calculate confidence interval
        std_payoffs = np.std(all_payoffs, ddof=1)  # Sample standard deviation
        n_total = len(all_payoffs)
        standard_error = std_payoffs / np.sqrt(n_total)
        
        # Get critical value for confidence interval
        from scipy import stats
        alpha = 1 - confidence_level
        critical_value = stats.norm.ppf(1 - alpha/2)
        
        # Calculate confidence interval for the discounted price
        margin_of_error = critical_value * standard_error * np.exp(-r * T)
        confidence_lower = option_price - margin_of_error
        confidence_upper = option_price + margin_of_error
        
        # Calculate additional statistics
        computation_time = time.time() - start_time
        
        # Monte Carlo error (theoretical standard error)
        mc_error = standard_error * np.exp(-r * T)
        
        # Calculate percentage of paths that hit the barrier
        barrier_hit_count = 0
        for path in all_paths:
            if 'down_and_out' in option_type or 'down_and_in' in option_type:
                if np.any(path <= B):
                    barrier_hit_count += 1
            else:  # up options
                if np.any(path >= B):
                    barrier_hit_count += 1
        
        barrier_hit_percentage = (barrier_hit_count / n_total) * 100
        
        statistics = {
            'mean_payoff': mean_payoff,
            'std_payoff': std_payoffs,
            'standard_error': standard_error,
            'mc_error': mc_error,
            'barrier_hit_percentage': barrier_hit_percentage,
            'computation_time': computation_time,
            'effective_simulations': n_total,
            'convergence_ratio': mc_error / option_price if option_price > 0 else float('inf')
        }
        
        return option_price, confidence_lower, confidence_upper, statistics
    
    def analyze_convergence(self, S0: float, K: float, B: float, T: float,
                           r: float, sigma: float, option_type: str,
                           N_steps: int, sim_counts: list = None,
                           monitoring_type: str = 'discrete') -> dict:
        """
        Analyze convergence of the Monte Carlo estimate as number of simulations increases.
        
        Parameters:
        -----------
        S0, K, B, T, r, sigma, option_type, N_steps : option parameters
        sim_counts : list
            List of simulation counts to test
        monitoring_type : str
            Monitoring type
        
        Returns:
        --------
        dict
            Dictionary containing convergence analysis results
        """
        if sim_counts is None:
            sim_counts = [1000, 5000, 10000, 50000, 100000, 500000]
        
        prices = []
        errors = []
        times = []
        
        for N_sim in sim_counts:
            price, _, _, stats = self.monte_carlo_pricer(
                S0, K, B, T, r, sigma, option_type, N_sim, N_steps, monitoring_type
            )
            prices.append(price)
            errors.append(stats['mc_error'])
            times.append(stats['computation_time'])
        
        return {
            'sim_counts': sim_counts,
            'prices': prices,
            'errors': errors,
            'computation_times': times
        }
    
    def sensitivity_analysis(self, S0: float, K: float, B: float, T: float,
                           r: float, sigma: float, option_type: str,
                           N_sim: int, N_steps: int,
                           parameter: str, range_pct: float = 0.1,
                           num_points: int = 11) -> dict:
        """
        Perform sensitivity analysis on a specified parameter.
        
        Parameters:
        -----------
        S0, K, B, T, r, sigma, option_type, N_sim, N_steps : option parameters
        parameter : str
            Parameter to vary ('S0', 'sigma', 'r', etc.)
        range_pct : float
            Percentage range around base value to explore
        num_points : int
            Number of points in the sensitivity analysis
        
        Returns:
        --------
        dict
            Dictionary containing sensitivity analysis results
        """
        base_params = {'S0': S0, 'K': K, 'B': B, 'T': T, 'r': r, 'sigma': sigma}
        base_value = base_params[parameter]
        
        # Create range of values
        min_val = base_value * (1 - range_pct)
        max_val = base_value * (1 + range_pct)
        param_values = np.linspace(min_val, max_val, num_points)
        
        prices = []
        
        for param_val in param_values:
            # Update the parameter
            current_params = base_params.copy()
            current_params[parameter] = param_val
            
            # Price the option
            price, _, _, _ = self.monte_carlo_pricer(
                current_params['S0'], current_params['K'], current_params['B'],
                current_params['T'], current_params['r'], current_params['sigma'],
                option_type, N_sim, N_steps
            )
            prices.append(price)
        
        return {
            'parameter': parameter,
            'values': param_values,
            'prices': prices,
            'base_value': base_value,
            'base_price': prices[num_points // 2]
        }

def print_detailed_results(price: float, conf_lower: float, conf_upper: float, 
                          stats: dict, option_params: dict):
    """Print detailed results of the barrier option pricing."""
    print("\n" + "="*60)
    print("BARRIER OPTION PRICING RESULTS")
    print("="*60)
    
    # Option details
    print(f"\nOption Type: {option_params['option_type'].replace('_', ' ').title()}")
    print(f"Initial Asset Price (S0): ${option_params['S0']:.2f}")
    print(f"Strike Price (K): ${option_params['K']:.2f}")
    print(f"Barrier Level (B): ${option_params['B']:.2f}")
    print(f"Time to Maturity (T): {option_params['T']:.4f} years")
    print(f"Risk-free Rate (r): {option_params['r']:.2%}")
    print(f"Volatility (σ): {option_params['sigma']:.2%}")
    
    # Simulation parameters
    print(f"\nSimulation Parameters:")
    print(f"Number of Simulations: {stats['effective_simulations']:,}")
    print(f"Number of Time Steps: {option_params['N_steps']:,}")
    print(f"Monitoring Type: {option_params.get('monitoring_type', 'discrete').title()}")
    
    # Results
    print(f"\n" + "-"*30)
    print("PRICING RESULTS")
    print("-"*30)
    print(f"Option Price: ${price:.6f}")
    print(f"95% Confidence Interval: [${conf_lower:.6f}, ${conf_upper:.6f}]")
    print(f"Monte Carlo Error: ±${stats['mc_error']:.6f}")
    print(f"Convergence Ratio: {stats['convergence_ratio']:.4f}")
    
    # Additional statistics
    print(f"\n" + "-"*30)
    print("ADDITIONAL STATISTICS")
    print("-"*30)
    print(f"Mean Payoff (undiscounted): ${stats['mean_payoff']:.6f}")
    print(f"Standard Deviation of Payoffs: ${stats['std_payoff']:.6f}")
    print(f"Barrier Hit Percentage: {stats['barrier_hit_percentage']:.2f}%")
    print(f"Computation Time: {stats['computation_time']:.3f} seconds")

def main():
    """Main function demonstrating the usage of the BarrierOptionsPricer."""
    # Initialize the pricer
    pricer = BarrierOptionsPricer()
    
    # Example parameters
    example_params = {
        'S0': 100.0,        # Initial asset price
        'K': 100.0,         # Strike price
        'B': 90.0,          # Barrier level
        'T': 1.0,           # Time to maturity (1 year)
        'r': 0.05,          # Risk-free rate (5%)
        'sigma': 0.2,       # Volatility (20%)
        'option_type': 'down_and_out_call',  # Option type
        'N_sim': 100000,    # Number of simulations
        'N_steps': 252,     # Number of time steps (daily monitoring)
        'monitoring_type': 'discrete'
    }
    
    print("Monte Carlo Barrier Options Pricer")
    print("===================================")
    
    # Option 1: Use predefined parameters
    use_default = input("Use default parameters? (y/n): ").lower().strip() == 'y'
    
    if not use_default:
        # Get user inputs
        print("\nEnter option parameters:")
        try:
            example_params['S0'] = float(input(f"Initial asset price (default {example_params['S0']}): ") or example_params['S0'])
            example_params['K'] = float(input(f"Strike price (default {example_params['K']}): ") or example_params['K'])
            example_params['B'] = float(input(f"Barrier level (default {example_params['B']}): ") or example_params['B'])
            example_params['T'] = float(input(f"Time to maturity in years (default {example_params['T']}): ") or example_params['T'])
            example_params['r'] = float(input(f"Risk-free rate (default {example_params['r']}): ") or example_params['r'])
            example_params['sigma'] = float(input(f"Volatility (default {example_params['sigma']}): ") or example_params['sigma'])
            
            print(f"\nAvailable option types: {pricer.valid_option_types}")
            option_type_input = input(f"Option type (default {example_params['option_type']}): ").strip()
            if option_type_input:
                example_params['option_type'] = option_type_input
            
            example_params['N_sim'] = int(input(f"Number of simulations (default {example_params['N_sim']}): ") or example_params['N_sim'])
            example_params['N_steps'] = int(input(f"Number of time steps (default {example_params['N_steps']}): ") or example_params['N_steps'])
            
            monitoring_input = input(f"Monitoring type - discrete/continuous_approx (default {example_params['monitoring_type']}): ").strip()
            if monitoring_input:
                example_params['monitoring_type'] = monitoring_input
        except ValueError as e:
            print(f"Invalid input: {e}. Using default parameters.")
    
    # Price the option
    print("\nPricing the barrier option...")
    price, conf_lower, conf_upper, stats = pricer.monte_carlo_pricer(
        example_params['S0'], example_params['K'], example_params['B'],
        example_params['T'], example_params['r'], example_params['sigma'],
        example_params['option_type'], example_params['N_sim'], example_params['N_steps'],
        example_params['monitoring_type']
    )
    
    # Print detailed results
    print_detailed_results(price, conf_lower, conf_upper, stats, example_params)
    
    # Optional: Perform additional analysis
    additional_analysis = input("\nPerform additional analysis? (y/n): ").lower().strip() == 'y'
    
    if additional_analysis:
        print("\n1. Convergence Analysis")
        print("2. Sensitivity Analysis")
        analysis_choice = input("Choose analysis type (1/2): ").strip()
        
        if analysis_choice == '1':
            print("\nPerforming convergence analysis...")
            convergence = pricer.analyze_convergence(
                example_params['S0'], example_params['K'], example_params['B'],
                example_params['T'], example_params['r'], example_params['sigma'],
                example_params['option_type'], example_params['N_steps'],
                monitoring_type=example_params['monitoring_type']
            )
            
            print("\nConvergence Analysis Results:")
            print("-" * 50)
            print(f"{'Simulations':<12} {'Price':<12} {'MC Error':<12} {'Time (s)':<10}")
            print("-" * 50)
            for i, n_sim in enumerate(convergence['sim_counts']):
                print(f"{n_sim:<12,} ${convergence['prices'][i]:<11.6f} ±{convergence['errors'][i]:<11.6f} {convergence['computation_times'][i]:<10.3f}")
        
        elif analysis_choice == '2':
            print("\nAvailable parameters for sensitivity analysis:")
            print("S0, K, B, T, r, sigma")
            param = input("Choose parameter: ").strip()
            
            if param in ['S0', 'K', 'B', 'T', 'r', 'sigma']:
                print(f"\nPerforming sensitivity analysis on {param}...")
                sensitivity = pricer.sensitivity_analysis(
                    example_params['S0'], example_params['K'], example_params['B'],
                    example_params['T'], example_params['r'], example_params['sigma'],
                    example_params['option_type'], example_params['N_sim']//5, 
                    example_params['N_steps'], param
                )
                
                print(f"\nSensitivity Analysis Results for {param}:")
                print("-" * 40)
                print(f"{'Value':<15} {'Option Price':<15}")
                print("-" * 40)
                for i, val in enumerate(sensitivity['values']):
                    print(f"{val:<15.6f} ${sensitivity['prices'][i]:<15.6f}")
            else:
                print("Invalid parameter choice.")

if __name__ == "__main__":
    # Import scipy.stats for confidence intervals
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not available. Using normal approximation for confidence intervals.")
        # Fallback implementation using normal approximation
        class stats:
            class norm:
                @staticmethod
                def ppf(x):
                    # Approximate inverse normal CDF for 95% confidence
                    if abs(x - 0.975) < 1e-6:
                        return 1.96
                    elif abs(x - 0.95) < 1e-6:
                        return 1.645
                    else:
                        # Simple approximation
                        return x * 2
    
    main()