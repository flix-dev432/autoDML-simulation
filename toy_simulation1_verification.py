import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def sigmoid(x):
    """Sigmoid function for probability calculation"""
    return 1 / (1 + np.exp(-x))

def generate_data(n_samples=1000000):
    """
    Generate data according to the specified DGP
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    X : numpy.ndarray
        Binary covariates (n_samples x 3)
    """
    # Correlation matrix for latent variables
    cov_matrix = np.array([
        [1.0, 0.4, 0.4],
        [0.4, 1.0, 0.4],
        [0.4, 0.4, 1.0]
    ])
    
    # Generate latent variables Z from multivariate normal
    Z = np.random.multivariate_normal(mean=[0, 0, 0], cov=cov_matrix, size=n_samples)
    
    # Binarize to get X
    X = (Z > 0).astype(int)
    
    return X

def calculate_target_parameter(X):
    """
    Calculate the theoretical target parameter Ψ₀ = E[1.2 + 0.5*X₁ - 0.7*X₂]
    
    Parameters:
    -----------
    X : numpy.ndarray
        Binary covariates (n_samples x 3)
        
    Returns:
    --------
    float
        Estimated target parameter
    """
    # True target parameter is Ψ₀ = E[1.2 + 0.5*X₁ - 0.7*X₂]
    individual_effects = 1.2 + 0.5 * X[:, 0] - 0.7 * X[:, 1]
    target_parameter = individual_effects.mean()
    
    return target_parameter

def verify_marginal_probabilities(X):
    """Verify the marginal probabilities of X"""
    marginal_probs = X.mean(axis=0)
    print("Marginal probabilities of X:")
    for j in range(3):
        print(f"E[X{j+1}] = {marginal_probs[j]:.4f} (expected: 0.5000)")
    
    return marginal_probs

def verify_pairwise_correlations(X):
    """Verify the pairwise correlations between X variables"""
    corr_matrix = np.corrcoef(X, rowvar=False)
    print("\nPairwise correlations of X:")
    for j in range(3):
        for k in range(j+1, 3):
            print(f"Corr(X{j+1}, X{k+1}) = {corr_matrix[j, k]:.4f} (expected: ~0.2600)")
    
    return corr_matrix

# This function is now handled directly in main() with the print_output function
def run_monte_carlo(n_samples=1000000, n_sims=10000, print_func=print):
    """
    Run Monte Carlo simulation to verify the target parameter
    
    Parameters:
    -----------
    n_samples : int
        Number of samples per simulation
    n_sims : int
        Number of Monte Carlo simulations
    print_func : function
        Function to use for printing output
        
    Returns:
    --------
    numpy.ndarray
        Array of target parameter estimates across simulations
    """
    target_params = np.zeros(n_sims)
    
    for i in range(n_sims):
        if i % 100 == 0:
            print_func(f"Running simulation {i+1}/{n_sims}")
        
        X = generate_data(n_samples)
        target_params[i] = calculate_target_parameter(X)
    
    return target_params

def main():
    # Open a file to save all outputs
    with open('simulation_results.txt', 'w') as f:
        # Function to print to both console and file
        def print_output(message):
            print(message)
            f.write(message + '\n')
        
        # Generate a single dataset for verification of data properties
        print_output("Generating a dataset for verification of properties...")
        X = generate_data(n_samples=1000000)
        
        # Verify the properties of X
        marginal_probs = X.mean(axis=0)
        print_output("Marginal probabilities of X:")
        for j in range(3):
            print_output(f"E[X{j+1}] = {marginal_probs[j]:.4f} (expected: 0.5000)")
        
        corr_matrix = np.corrcoef(X, rowvar=False)
        print_output("\nPairwise correlations of X:")
        for j in range(3):
            for k in range(j+1, 3):
                print_output(f"Corr(X{j+1}, X{k+1}) = {corr_matrix[j, k]:.4f} (expected: ~0.2600)")
        
        # Calculate target parameter for this dataset
        target_param = calculate_target_parameter(X)
        print_output(f"\nTarget parameter Ψ₀ = E[1.2 + 0.5*X₁ - 0.7*X₂] = {target_param:.4f}")
        print_output(f"Theoretical value: Ψ₀ = 1.2 + 0.5*0.5 - 0.7*0.5 = 1.1000")
        
        # Run Monte Carlo simulation
        print_output("\nRunning Monte Carlo simulation to verify target parameter...")
        
        # Modify run_monte_carlo to use print_output
        target_params = np.zeros(10000)  # n_sims=10000
        for i in range(10000):
            if i % 1000 == 0:
                print_output(f"Running simulation {i+1}/10000")
            X = generate_data(1000000)  # n_samples=1000000
            target_params[i] = calculate_target_parameter(X)
        
        mc_results = target_params
        
        # Analyze Monte Carlo results
        mc_mean = mc_results.mean()
        mc_std = mc_results.std()
        mc_ci_lower = np.percentile(mc_results, 2.5)
        mc_ci_upper = np.percentile(mc_results, 97.5)
        
        print_output("\nMonte Carlo Results:")
        print_output(f"Mean of target parameter: {mc_mean:.4f}")
        print_output(f"Standard deviation: {mc_std:.4f}")
        print_output(f"95% CI: [{mc_ci_lower:.4f}, {mc_ci_upper:.4f}]")
        print_output(f"Deviation from theoretical value (1.1): {mc_mean - 1.1:.4f}")
        
        # Plot the distribution of target parameter estimates
        plt.figure(figsize=(10, 6))
        sns.histplot(mc_results, kde=True)
        plt.axvline(1.1, color='red', linestyle='--', label='Theoretical Ψ₀ = 1.1')
        plt.axvline(mc_mean, color='blue', linestyle='-', label=f'Monte Carlo Mean = {mc_mean:.4f}')
        plt.title('Distribution of Target Parameter Estimates')
        plt.xlabel('Target Parameter Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('target_parameter_distribution.png')
        print_output("\nDistribution plot saved as 'target_parameter_distribution.png'")
        print_output("\nAll results have been saved to 'simulation_results.txt'")

if __name__ == "__main__":
    main()