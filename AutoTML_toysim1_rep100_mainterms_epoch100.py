import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
import multiprocessing as mp
from joblib import Parallel, delayed
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages

device = torch.device("cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Data generation function
def sigmoid(x):
    """Sigmoid function for probability calculation"""
    return 1 / (1 + np.exp(-x))

def generate_data(n_samples):
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
    A : numpy.ndarray
        Binary treatment assignment
    Y : numpy.ndarray
        Binary outcome
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
    
    # Compute propensity score: logit(P(A=1|X)) = 0.5 - 1.5*X1 + 0.5*X2
    logit_pA = 0.5 - 1.5 * X[:, 0] + 0.5 * X[:, 1]
    pA = sigmoid(logit_pA)
    A = (np.random.random(n_samples) < pA).astype(int)
    
    # Compute outcome probability
    # logit(P(Y=1|A,X)) = -1.386 + 1.2*A + 0.5*X1 + 0.5*X2 + 0.5*X3 + 0.5*A*X1 - 0.7*A*X2
    logit_pY = -1.386 + 1.2 * A + 0.5 * X[:, 0] + 0.5 * X[:, 1] + 0.5 * X[:, 2] + 0.5 * A * X[:, 0] - 0.7 * A * X[:, 1]
    pY = sigmoid(logit_pY)
    Y = (np.random.random(n_samples) < pY).astype(int)
    
    return X, A, Y

# Calculate true target parameter for this dataset
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

# Linear Model for theta
class ThetaNet(nn.Module):
    def __init__(self, input_dim):
        super(ThetaNet, self).__init__()
        
        # Simple linear layer for b0 + b1*A + b2*X1 + b3*X2 + b4*X3
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, a, x):
        # Concatenate treatment A and covariates X
        inputs = torch.cat([a.unsqueeze(1), x], dim=1)
        return self.linear(inputs)

# Linear Model for alpha components
class AlphaComponentNet(nn.Module):
    def __init__(self, input_dim):
        super(AlphaComponentNet, self).__init__()
        
        # Replace neural network with simple linear layer: b0 + b1X1 + b2X2 + b3X3
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Alpha Model combining the two components
class AlphaNet:
    def __init__(self, input_dim):
        self.g0 = AlphaComponentNet(input_dim).to(device)  # For a=0
        self.g1 = AlphaComponentNet(input_dim).to(device)  # For a=1
    
    def __call__(self, a, x):
        """
        Compute alpha(a, x) = a*g1(x) + (1-a)*g0(x)
        """
        g0_output = self.g0(x)
        g1_output = self.g1(x)
        
        # Use broadcasting for element-wise calculation
        return a.unsqueeze(1) * g1_output + (1 - a.unsqueeze(1)) * g0_output
    
    def parameters(self):
        """Return all parameters for optimization"""
        return list(self.g0.parameters()) + list(self.g1.parameters())

# Train theta model
def train_theta(theta_net, a, x, y, lambda_theta=0.001, epochs=100, batch_size=64, lr=0.01, 
               patience=10, convergence_threshold=1e-4):
    """Train the theta neural network"""
    optimizer = optim.Adam(theta_net.parameters(), lr=lr)
    
    # Create dataset and dataloader
    dataset = TensorDataset(a, x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # For early stopping
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    
    # Training loop
    theta_net.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_a, batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Get theta predictions
            theta_pred = theta_net(batch_a, batch_x).squeeze()
            
            # Negative log-likelihood (Binary cross-entropy)
            neg_log_likelihood = -batch_y * theta_pred + torch.log(torch.exp(theta_pred) + 1)
            
            # L2 regularization
            l2_reg = 0
            for param in theta_net.parameters():
                l2_reg += param.norm(2)
            
            # Total loss
            loss = neg_log_likelihood.mean() + lambda_theta * l2_reg
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        
        # Early stopping check
        if avg_epoch_loss < best_loss - convergence_threshold:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Check for convergence
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}/{epochs} with loss {avg_epoch_loss:.6f}")
            break
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
    
    # Check if training completed all epochs or stopped early
    if epoch == epochs - 1:
        print(f"Completed all {epochs} epochs with final loss: {avg_epoch_loss:.6f}")
    
    # Return model and training losses
    return theta_net, losses

# Implementation of autoTML algorithm
def autoTML(X, A, Y, n_splits=5, lambda_theta=0.001, lambda_alpha=0.001, theta_epochs=100, alpha_epochs=100, 
           batch_size=64, lr_theta=0.01, lr_alpha=0.01, seed=None):
    """
    Implement the autoTML algorithm with cross-fitting
    
    Parameters:
    -----------
    X : numpy.ndarray
        Binary covariates (n_samples x 3)
    A : numpy.ndarray
        Binary treatment assignment
    Y : numpy.ndarray
        Binary outcome
    n_splits : int
        Number of cross-fitting splits
    lambda_theta, lambda_alpha : float
        Regularization parameters
    theta_epochs, alpha_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr_theta, lr_alpha : float
        Learning rates
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with estimation results
    """
    # Set seed if provided (for parallel runs)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    n = len(X)
    input_dim = X.shape[1]
    
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X).to(device)
    A_tensor = torch.FloatTensor(A).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)
    
    # Create data splits for cross-fitting
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, n_splits)
    
    # Maps each sample to its fold
    fold_mapping = np.zeros(n, dtype=int)
    for j, fold_indices in enumerate(split_indices):
        fold_mapping[fold_indices] = j
    
    # Initialize lists to store cross-fitted models
    theta_models = []
    alpha_models = []
    
    # Step 1: Cross-fit theta models
    print("Step 1: Cross-fitting theta models...")
    for s in range(n_splits):
        print(f"Training theta model for fold {s+1}/{n_splits}")
        
        # Get train and test indices for this split
        test_indices = split_indices[s]
        train_indices = np.concatenate([split_indices[j] for j in range(n_splits) if j != s])
        
        # Training data for this split
        X_train = X_tensor[train_indices]
        A_train = A_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        
        # Train theta model on training data
        theta_net = ThetaNet(input_dim + 1).to(device)  # +1 for treatment A
        theta_net, _ = train_theta(theta_net, A_train, X_train, Y_train, 
                               lambda_theta=lambda_theta, epochs=theta_epochs, 
                               batch_size=batch_size, lr=lr_theta)
        
        # Store theta model
        theta_models.append(theta_net)
    
    # Step 2: Cross-fit alpha models using pre-trained theta models
    print("Step 2: Cross-fitting alpha models...")
    for s in range(n_splits):
        print(f"Training alpha model for fold {s+1}/{n_splits}")
        
        # Get train and test indices for this split
        test_indices = split_indices[s]
        train_indices = np.concatenate([split_indices[j] for j in range(n_splits) if j != s])
        
        # Initialize alpha model
        alpha_net = AlphaNet(input_dim)
        optimizer = optim.Adam(alpha_net.parameters(), lr=lr_alpha)
        
        # Create a mapping from training sample to its original index
        train_to_original = {i: train_indices[i] for i in range(len(train_indices))}
        
        # Create dataset for training
        train_dataset = TensorDataset(torch.arange(len(train_indices)), 
                                     A_tensor[train_indices], 
                                     X_tensor[train_indices])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        alpha_net.g0.train()
        alpha_net.g1.train()
        
        # For early stopping
        best_loss = float('inf')
        patience_counter = 0
        alpha_losses = []
        
        for epoch in range(alpha_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_local_idx, batch_a, batch_x in train_dataloader:
                optimizer.zero_grad()
                
                # Get original indices for this batch
                orig_indices = [train_to_original[idx.item()] for idx in batch_local_idx]
                
                # Get the folds for these samples
                batch_folds = fold_mapping[orig_indices]
                
                # Initialize batch loss
                batch_loss = 0
                
                # Process samples grouped by fold to minimize model switching
                unique_folds = np.unique(batch_folds)
                for fold in unique_folds:
                    # Get samples for this fold
                    fold_mask = (batch_folds == fold)
                    fold_idxs = np.where(fold_mask)[0]
                    
                    # Get data for this fold
                    fold_a = batch_a[fold_mask]
                    fold_x = batch_x[fold_mask]
                    
                    # Get theta model trained WITHOUT this fold
                    theta_net = theta_models[fold]
                    theta_net.eval()
                    
                    # Calculate alpha loss using the corresponding theta model
                    with torch.no_grad():
                        # Get theta predictions
                        theta_axs = theta_net(fold_a, fold_x).squeeze()
                        
                        # Calculate sigmoid derivatives: sigmoid(x)*(1-sigmoid(x))
                        sigmoid_thetas = torch.sigmoid(theta_axs)
                        sigmoid_derivs = sigmoid_thetas * (1 - sigmoid_thetas)
                    
                    # Alpha for current (a,x) pairs
                    alpha_axs = alpha_net(fold_a, fold_x).squeeze()
                    
                    # Compute alpha(1,x) and alpha(0,x) for all x in this fold
                    ones = torch.ones_like(fold_a)
                    zeros = torch.zeros_like(fold_a)
                    
                    alpha_1xs = alpha_net.g1(fold_x).squeeze()
                    alpha_0xs = alpha_net.g0(fold_x).squeeze()
                    alpha_diffs = alpha_1xs - alpha_0xs
                    
                    # Loss terms (vectorized)
                    first_terms = sigmoid_derivs * alpha_axs**2
                    second_terms = -2 * alpha_diffs
                    
                    # Add to batch loss
                    fold_losses = first_terms + second_terms
                    batch_loss += fold_losses.sum()
                
                # Average loss over batch
                batch_loss = batch_loss / len(batch_local_idx)
                
                # Add L2 regularization
                l2_reg = 0
                for param in alpha_net.parameters():
                    l2_reg += param.norm(2)
                
                batch_loss += lambda_alpha * l2_reg
                
                # Backward and optimize
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
            
            # Average loss for this epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
            alpha_losses.append(avg_epoch_loss)
            
            # Early stopping check (with patience and convergence threshold)
            if avg_epoch_loss < best_loss - 1e-4:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Check for convergence
            if patience_counter >= 10:  # 10 epochs of patience
                print(f"Alpha model: Early stopping at epoch {epoch+1}/{alpha_epochs} with loss {avg_epoch_loss:.6f}")
                break
                
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Alpha model: Epoch {epoch+1}/{alpha_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Check if training completed all epochs
        if epoch == alpha_epochs - 1:
            print(f"Alpha model: Completed all {alpha_epochs} epochs with final loss: {avg_epoch_loss:.6f}")
        
        # Store alpha model and losses
        alpha_models.append((alpha_net, alpha_losses))
    
    # Step 3: Targeting step - compute epsilon_n
    # Collect theta and alpha predictions for all samples
    print("Step 3: Computing targeting step (epsilon_n)...")
    theta_preds = np.zeros(n)
    alpha_preds = np.zeros(n)
    
    for i in range(n):
        # Get the model fold for this sample
        fold = fold_mapping[i]
        
        # Get models trained without this sample
        theta_net = theta_models[fold]
        alpha_net = alpha_models[fold][0] if isinstance(alpha_models[fold], tuple) else alpha_models[fold]
        
        # Get data for this sample
        x_i = X_tensor[i:i+1]
        a_i = A_tensor[i:i+1]
        
        with torch.no_grad():
            # Predict theta(A,X) and alpha(A,X)
            theta_preds[i] = theta_net(a_i, x_i).item()
            alpha_preds[i] = alpha_net(a_i, x_i).item()
    
    # Define negative log-likelihood loss function for epsilon optimization
    def neg_log_likelihood_loss(epsilon):
        # θ* = θ + ε·α
        theta_star = theta_preds + epsilon * alpha_preds
        
        # -Y·θ*(A,X) + log(1 + exp(θ*(A,X)))
        neg_ll = -Y * theta_star + np.log(1 + np.exp(theta_star))
        return np.sum(neg_ll)

    # Optimize epsilon using scipy
    result = minimize(neg_log_likelihood_loss, x0=0, method='BFGS')
    epsilon_n = result.x[0]
    print(f"  Computed epsilon_n = {epsilon_n:.6f}")
    
    # Step 4: Update theta to theta* = theta + epsilon_n·alpha
    print("Step 4: Computing plug-in estimator...")
    
    # Initialize arrays to store efficient influence function values and EIF terms
    eif_values = np.zeros(n)
    eif_terms = np.zeros(n)
    
    # Compute plug-in estimator
    sum_term = 0
    for i in range(n):
        # Get the model fold for this sample
        fold = fold_mapping[i]
        
        # Get models trained without this sample
        theta_net = theta_models[fold]
        alpha_net = alpha_models[fold][0] if isinstance(alpha_models[fold], tuple) else alpha_models[fold]
        
        # Get data for this sample
        x_i = X_tensor[i:i+1]
        a_i = A_tensor[i:i+1]
        y_i = Y_tensor[i:i+1].item()
        
        with torch.no_grad():
            # Original theta predictions
            theta_a_x = theta_net(a_i, x_i).item()
            
            # Alpha prediction
            alpha_a_x = alpha_net(a_i, x_i).item()
            
            # Compute theta* = theta + epsilon_n·alpha
            theta_star_a_x = theta_a_x + epsilon_n * alpha_a_x
            
            # Predictions for a=1 and a=0
            theta_star_1_x = theta_net(torch.ones_like(a_i), x_i).item() + epsilon_n * alpha_net(torch.ones_like(a_i), x_i).item()
            theta_star_0_x = theta_net(torch.zeros_like(a_i), x_i).item() + epsilon_n * alpha_net(torch.zeros_like(a_i), x_i).item()
            
            # Plug-in estimate contribution: theta*(1,X) - theta*(0,X)
            plug_in_term = theta_star_1_x - theta_star_0_x
            
            # Store for the final estimator
            sum_term += plug_in_term
            
            # Calculate efficient influence function component: (sigma(theta*(A,X)) - Y)·alpha(A,X)
            sigma_theta_star = sigmoid(theta_star_a_x)
            eif_term = (sigma_theta_star - y_i) * alpha_a_x
            
            # Store EIF terms for reporting
            eif_terms[i] = eif_term
            
            # Store EIF value for variance calculation
            eif_values[i] = plug_in_term - eif_term
    
    # Final plug-in estimate
    psi_hat = sum_term / n
    
    # Calculate average EIF term (should be close to zero for well-specified models)
    avg_eif_term = np.mean(eif_terms)
    
    # Step 5: Compute standard error
    print("Step 5: Computing standard error...")
    
    # Calculate variance using the EIF values
    centered_eif = eif_values - psi_hat
    V_hat = np.mean(centered_eif**2)
    
    # Standard error
    se_hat = np.sqrt(V_hat / n)
    
    # 95% confidence interval
    ci_lower = psi_hat - 1.96 * se_hat
    ci_upper = psi_hat + 1.96 * se_hat
    
    # Calculate average EIF value (should be close to 0 for well-specified models)
    avg_eif = np.mean(eif_values - psi_hat)
    
    # Return results
    results = {
        'psi_hat': psi_hat,
        'se_hat': se_hat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'variance': V_hat,
        'epsilon_n': epsilon_n,
        'avg_eif': avg_eif,
        'avg_eif_term': avg_eif_term,
        'eif_values': eif_values,
        'eif_terms': eif_terms,
        'theta_losses': [losses for _, losses in theta_models] if isinstance(theta_models[0], tuple) else [],
        'alpha_losses': [losses for _, losses in alpha_models] if isinstance(alpha_models[0], tuple) else []
    }
    
    # Optionally plot the convergence of models
    if isinstance(theta_models[0], tuple) or isinstance(alpha_models[0], tuple):
        # Create directory if it doesn't exist
        os.makedirs('convergence_plots', exist_ok=True)
        
        # Create plot file
        plot_filename = f'convergence_plots/convergence_n{n}_seed{seed}.pdf'
        with PdfPages(plot_filename) as pdf:
            # Plot theta model losses
            if isinstance(theta_models[0], tuple):
                plt.figure(figsize=(10, 6))
                for i, (_, losses) in enumerate(theta_models):
                    plt.plot(losses, label=f'Fold {i+1}')
                plt.title('Theta Model Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()
            
            # Plot alpha model losses
            if isinstance(alpha_models[0], tuple):
                plt.figure(figsize=(10, 6))
                for i, (_, losses) in enumerate(alpha_models):
                    plt.plot(losses, label=f'Fold {i+1}')
                plt.title('Alpha Model Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()
                
        print(f"Convergence plots saved to {plot_filename}")
    
    return results

# Process a single repetition
def process_single_rep(rep, size, true_psi, n_splits, lambda_theta, lambda_alpha, plot_convergence=True):
    """Process a single repetition of the simulation"""
    start_time = time.time()
    print(f"  Repetition {rep+1} for sample size {size}")
    
    # Generate data
    X, A, Y = generate_data(size)
    
    # Run autoTML with a unique seed for each repetition
    seed = 42 * size + rep  # Unique seed based on size and repetition
    res = autoTML(X, A, Y, n_splits=n_splits, lambda_theta=lambda_theta, lambda_alpha=lambda_alpha,
                 theta_epochs=100, alpha_epochs=100, seed=seed)
    
    # Calculate metrics for this repetition
    bias = res['psi_hat'] - true_psi
    bias_se_ratio = bias / res['se_hat'] if res['se_hat'] > 0 else float('inf')
    covered = res['ci_lower'] <= true_psi <= res['ci_upper']
    ci_width = res['ci_upper'] - res['ci_lower']
    
    end_time = time.time()
    
    # Create a report for this repetition
    if plot_convergence and (len(res.get('theta_losses', [])) > 0 or len(res.get('alpha_losses', [])) > 0):
        # Create directory if it doesn't exist
        os.makedirs(f'results/repetition_reports', exist_ok=True)
        
        report_filename = f'results/repetition_reports/rep_{rep+1}_size_{size}.txt'
        with open(report_filename, 'w') as f:
            f.write(f"Repetition {rep+1} for sample size {size}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Psi estimate: {res['psi_hat']:.6f}\n")
            f.write(f"Standard error: {res['se_hat']:.6f}\n")
            f.write(f"95% CI: [{res['ci_lower']:.6f}, {res['ci_upper']:.6f}]\n")
            f.write(f"Bias: {bias:.6f}\n")
            f.write(f"Bias/SE ratio: {bias_se_ratio:.6f}\n")
            f.write(f"Coverage: {covered}\n")
            f.write(f"CI width: {ci_width:.6f}\n")
            f.write(f"Epsilon_n: {res['epsilon_n']:.6f}\n")
            f.write(f"Average EIF: {res['avg_eif']:.6f}\n")
            f.write(f"Average EIF term: {res['avg_eif_term']:.6f}\n")
            f.write(f"Time taken: {end_time - start_time:.2f}s\n")
    
    return {
        'psi_hat': res['psi_hat'],
        'se_hat': res['se_hat'],
        'ci_lower': res['ci_lower'],
        'ci_upper': res['ci_upper'],
        'bias': bias,
        'bias_se_ratio': bias_se_ratio,
        'covered': covered,
        'ci_width': ci_width,
        'time': end_time - start_time,
        'epsilon_n': res['epsilon_n'],
        'avg_eif': res['avg_eif'],
        'avg_eif_term': res['avg_eif_term']
    }

def run_simulation(sample_sizes=[1000, 5000, 10000], n_reps=100, n_splits=5, 
                  lambda_theta=0.001, lambda_alpha=0.001, n_jobs=-1):
    """
    Run simulation for different sample sizes
    
    Parameters:
    -----------
    sample_sizes : list
        List of sample sizes to simulate
    n_reps : int
        Number of simulation repetitions
    n_splits : int
        Number of cross-fitting splits
    lambda_theta, lambda_alpha : float
        Regularization parameters
    n_jobs : int
        Number of parallel jobs to run (-1 for all available cores)
        
    Returns:
    --------
    dict
        Dictionary with simulation results
    """
    # True target parameter
    true_psi = 1.1
    
    # Results storage
    results = {size: {'psi_hats': [], 'ci_lowers': [], 'ci_uppers': [], 'ses': [], 
                      'biases': [], 'bias_se_ratios': [], 'coverages': [], 'ci_widths': [],
                      'epsilon_ns': [], 'avg_eifs': [], 'avg_eif_terms': []} 
               for size in sample_sizes}
    
    # Create directory for results
    os.makedirs('results', exist_ok=True)
    
    # Initialize results file
    with open('results/autotml_results.txt', 'w') as f:
        f.write("AutoTML Simulation Results\n")
        f.write("=========================\n\n")
        f.write(f"Number of repetitions: {n_reps}\n")
        f.write(f"Number of cross-fitting splits: {n_splits}\n")
        f.write(f"Regularization: lambda_theta={lambda_theta}, lambda_alpha={lambda_alpha}\n")
        f.write(f"True target parameter: {true_psi}\n\n")
        f.write("Using targeting step with epsilon_n optimization\n\n")
    
    # Set up parallel processing
    n_cores = mp.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Using {n_cores} cores for parallel processing")
    
    # Run simulation for each sample size
    for size in sample_sizes:
        print(f"\n\nSimulating for sample size {size}")
        
        # Process repetitions in parallel
        rep_results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_rep)(rep, size, true_psi, n_splits, lambda_theta, lambda_alpha)
            for rep in range(n_reps)
        )
        
        # Extract results
        for rep, res in enumerate(rep_results):
            # Store results
            results[size]['psi_hats'].append(res['psi_hat'])
            results[size]['ci_lowers'].append(res['ci_lower'])
            results[size]['ci_uppers'].append(res['ci_upper'])
            results[size]['ses'].append(res['se_hat'])
            results[size]['biases'].append(res['bias'])
            results[size]['bias_se_ratios'].append(res['bias_se_ratio'])
            results[size]['coverages'].append(res['covered'])
            results[size]['ci_widths'].append(res['ci_width'])
            results[size]['epsilon_ns'].append(res['epsilon_n'])
            results[size]['avg_eifs'].append(res['avg_eif'])
            results[size]['avg_eif_terms'].append(res['avg_eif_term'])
            
            # Write results to file
            with open('results/autotml_results.txt', 'a') as f:
                f.write(f"Sample size {size}, Rep {rep+1}: ")
                f.write(f"psi_hat={res['psi_hat']:.4f}, ")
                f.write(f"SE={res['se_hat']:.4f}, ")
                f.write(f"95% CI=[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}], ")
                f.write(f"bias={res['bias']:.4f}, ")
                f.write(f"bias/SE={res['bias_se_ratio']:.4f}, ")
                f.write(f"covered={res['covered']}, ")
                f.write(f"CI width={res['ci_width']:.4f}, ")
                f.write(f"epsilon_n={res['epsilon_n']:.6f}, ")
                f.write(f"avg_eif={res['avg_eif']:.6f}, ")
                f.write(f"avg_eif_term={res['avg_eif_term']:.6f}, ")
                f.write(f"time={res['time']:.2f}s\n")
        
        # Calculate and report aggregate metrics for this sample size
        psi_hats = np.array(results[size]['psi_hats'])
        ci_lowers = np.array(results[size]['ci_lowers'])
        ci_uppers = np.array(results[size]['ci_uppers'])
        ses = np.array(results[size]['ses'])
        biases = np.array(results[size]['biases'])
        bias_se_ratios = np.array(results[size]['bias_se_ratios'])
        epsilon_ns = np.array(results[size]['epsilon_ns'])
        avg_eifs = np.array(results[size]['avg_eifs'])
        avg_eif_terms = np.array(results[size]['avg_eif_terms'])
        
        mean_psi_hat = np.mean(psi_hats)
        mean_bias = np.mean(biases)
        mean_se = np.mean(ses)
        mean_bias_se_ratio = np.mean(bias_se_ratios)
        rmse = np.sqrt(np.mean((psi_hats - true_psi)**2))
        coverage = np.mean(results[size]['coverages'])
        mean_ci_width = np.mean(results[size]['ci_widths'])
        mean_epsilon_n = np.mean(epsilon_ns)
        std_epsilon_n = np.std(epsilon_ns)
        mean_avg_eif = np.mean(avg_eifs)
        mean_avg_eif_term = np.mean(avg_eif_terms)
        
        print(f"\nResults for sample size {size}:")
        print(f"  Mean estimate: {mean_psi_hat:.4f}")
        print(f"  Mean bias: {mean_bias:.4f}")
        print(f"  Mean standard error: {mean_se:.4f}")
        print(f"  Mean bias/SE ratio: {mean_bias_se_ratio:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Coverage: {coverage:.4f}")
        print(f"  Mean CI width: {mean_ci_width:.4f}")
        print(f"  Mean epsilon_n: {mean_epsilon_n:.6f} (SD: {std_epsilon_n:.6f})")
        print(f"  Mean avg EIF: {mean_avg_eif:.6f}")
        print(f"  Mean avg EIF term: {mean_avg_eif_term:.6f}")
        
        # Write aggregate results to file
        with open('results/autotml_results.txt', 'a') as f:
            f.write(f"\nAggregate results for sample size {size}:\n")
            f.write(f"  Mean estimate: {mean_psi_hat:.4f}\n")
            f.write(f"  Mean bias: {mean_bias:.4f}\n")
            f.write(f"  Mean standard error: {mean_se:.4f}\n")
            f.write(f"  Mean bias/SE ratio: {mean_bias_se_ratio:.4f}\n")
            f.write(f"  RMSE: {rmse:.4f}\n")
            f.write(f"  Coverage: {coverage:.4f}\n")
            f.write(f"  Mean CI width: {mean_ci_width:.4f}\n")
            f.write(f"  Mean epsilon_n: {mean_epsilon_n:.6f} (SD: {std_epsilon_n:.6f})\n")
            f.write(f"  Mean avg EIF: {mean_avg_eif:.6f}\n")
            f.write(f"  Mean avg EIF term: {mean_avg_eif_term:.6f}\n\n")
    
    # Create visualization of results
    plt.figure(figsize=(18, 15))
    
    # Plot 1: Estimates and CIs
    plt.subplot(3, 2, 1)
    for i, size in enumerate(sample_sizes):
        plt.errorbar(x=[i], y=[np.mean(results[size]['psi_hats'])], 
                    yerr=[np.mean(results[size]['ses']) * 1.96], 
                    fmt='o', capsize=5, label=f"n={size}")
    plt.axhline(y=true_psi, color='r', linestyle='--', label='True value')
    plt.xticks(range(len(sample_sizes)), [str(size) for size in sample_sizes])
    plt.ylabel('Estimate with 95% CI')
    plt.xlabel('Sample Size')
    plt.title('Mean Estimates with 95% CIs')
    plt.legend()
    
    # Plot 2: Mean Bias
    plt.subplot(3, 2, 2)
    biases = [np.mean(results[size]['biases']) for size in sample_sizes]
    plt.bar(range(len(sample_sizes)), biases)
    plt.xticks(range(len(sample_sizes)), [str(size) for size in sample_sizes])
    plt.ylabel('Bias')
    plt.xlabel('Sample Size')
    plt.title('Mean Bias')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Plot 3: Coverage
    plt.subplot(3, 2, 3)
    coverages = [np.mean(results[size]['coverages']) for size in sample_sizes]
    plt.bar(range(len(sample_sizes)), coverages)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% target')
    plt.xticks(range(len(sample_sizes)), [str(size) for size in sample_sizes])
    plt.ylim(0, 1)
    plt.ylabel('Coverage')
    plt.xlabel('Sample Size')
    plt.title('Coverage of 95% CI')
    plt.legend()
    
    # Plot 4: Distribution of estimates
    plt.subplot(3, 2, 4)
    for size in sample_sizes:
        sns.kdeplot(results[size]['psi_hats'], label=f"n={size}")
    plt.axvline(x=true_psi, color='r', linestyle='--', label='True value')
    plt.xlabel('Estimate')
    plt.ylabel('Density')
    plt.title('Distribution of Estimates')
    plt.legend()
    
    # Plot 5: Bias to SE ratio
    plt.subplot(3, 2, 5)
    bias_se_ratios = [np.mean(results[size]['bias_se_ratios']) for size in sample_sizes]
    plt.bar(range(len(sample_sizes)), bias_se_ratios)
    plt.xticks(range(len(sample_sizes)), [str(size) for size in sample_sizes])
    plt.ylabel('Bias/SE Ratio')
    plt.xlabel('Sample Size')
    plt.title('Mean Bias to Standard Error Ratio')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Plot 6: Distribution of bias/SE ratios
    plt.subplot(3, 2, 6)
    for size in sample_sizes:
        sns.kdeplot(results[size]['bias_se_ratios'], label=f"n={size}")
    plt.axvline(x=0, color='r', linestyle='--', label='Zero bias')
    plt.xlabel('Bias/SE Ratio')
    plt.ylabel('Density')
    plt.title('Distribution of Bias/SE Ratios')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/autotml_results.png')
    
    return results

if __name__ == "__main__":
    # Set hyperparameters
    n_reps = 100  # Number of simulation repetitions
    n_splits = 5  # Number of cross-fitting splits
    lambda_theta = 0.001  # Regularization for theta
    lambda_alpha = 0.001  # Regularization for alpha
    
    print("\nVerifying the theoretical target parameter...")
    # First verify the theoretical target parameter
    n_samples = 1000000
    X, _, _ = generate_data(n_samples)
    target_param = calculate_target_parameter(X)
    print(f"Target parameter with {n_samples} samples: {target_param:.4f}")
    print(f"Theoretical value: 1.1000")
    
    print("\nRunning autoTML simulation...")
    # Run the simulation with specified sample sizes
    results = run_simulation(
        sample_sizes=[1000, 5000, 10000, 30000, 50000],
        n_reps=n_reps,
        n_splits=n_splits,
        lambda_theta=lambda_theta,
        lambda_alpha=lambda_alpha,
        n_jobs=-1  # Use all available cores
    )
    
    print("\nSimulation complete. Results saved to 'results/autotml_results.txt' and 'results/autotml_results.png'.")
    