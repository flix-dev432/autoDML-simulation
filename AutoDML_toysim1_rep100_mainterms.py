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
class LinearTheta(nn.Module):
    def __init__(self, input_dim):
        super(LinearTheta, self).__init__()
        # Simple linear layer for b0 + b1*A + b2*X1 + b3*X2 + b4*X3
        self.linear = nn.Linear(input_dim + 1, 1)  # +1 for treatment A
    
    def forward(self, a, x):
        # Concatenate treatment A and covariates X
        inputs = torch.cat([a.unsqueeze(1), x], dim=1)
        return self.linear(inputs)

# Linear Model for alpha components
class LinearAlphaComponent(nn.Module):
    def __init__(self, input_dim):
        super(LinearAlphaComponent, self).__init__()
        # Simple linear layer for b0 + b1*X1 + b2*X2 + b3*X3
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Alpha Model combining the two components
class LinearAlpha:
    def __init__(self, input_dim):
        self.g0 = LinearAlphaComponent(input_dim).to(device)  # For a=0
        self.g1 = LinearAlphaComponent(input_dim).to(device)  # For a=1
    
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
def train_theta(theta_model, a, x, y, lambda_theta=0.001, epochs=200, batch_size=64, lr=0.01, 
               conv_threshold=1e-5, patience=5):
    """Train the theta linear model"""
    optimizer = optim.Adam(theta_model.parameters(), lr=lr) 
    
    # Add learning rate scheduler to dynamically adjust learning rate based on validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                   patience=3, min_lr=1e-5)
    
    # Create dataset and dataloader
    dataset = TensorDataset(a, x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    theta_model.train()
    prev_epoch_loss = float('inf')
    patience_counter = 0
    best_loss = float('inf')
    best_model_state = None
    prev_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_a, batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Get theta predictions - vectorized processing
            theta_pred = theta_model(batch_a, batch_x).squeeze()
            
            # Negative log-likelihood (Binary cross-entropy) 
            neg_log_likelihood = -batch_y * theta_pred + torch.log(torch.exp(theta_pred) + 1)
            
            # L2 regularization
            l2_reg = 0
            for param in theta_model.parameters():
                l2_reg += param.norm(2)
            
            # Total loss
            loss = neg_log_likelihood.mean() + lambda_theta * l2_reg
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss
        avg_epoch_loss = epoch_loss / num_batches
        
        # Update learning rate scheduler
        scheduler.step(avg_epoch_loss)
        
        # Track learning rate changes 
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            prev_lr = current_lr
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = theta_model.state_dict().copy()
        
        # Early stopping check
        if abs(prev_epoch_loss - avg_epoch_loss) < conv_threshold:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} with loss change: {abs(prev_epoch_loss - avg_epoch_loss):.8f}")
                break
        else:
            patience_counter = 0
            
        prev_epoch_loss = avg_epoch_loss
    
    # Restore best model state
    if best_model_state is not None:
        theta_model.load_state_dict(best_model_state)
    
    return theta_model

# Implementation of autoDML algorithm (modified to use linear models)
def autoDML(X, A, Y, n_splits=5, lambda_theta=0.001, lambda_alpha=0.001, theta_epochs=200, alpha_epochs=200, 
            batch_size=64, lr_theta=0.01, lr_alpha=0.01, stabilization=True, seed=None,
            conv_threshold=1e-5, patience=5):
    """
    Implement the autoDML algorithm with cross-fitting using linear models
    
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
    stabilization : bool
        Whether to apply TMLE-inspired stabilization
    seed : int, optional
        Random seed for reproducibility
    conv_threshold : float
        Convergence threshold for early stopping
    patience : int
        Number of epochs with small changes before early stopping
        
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
        
        # Train theta model on training data - using linear model
        theta_model = LinearTheta(input_dim).to(device)  # Using LinearTheta instead of ThetaNet
        theta_model = train_theta(theta_model, A_train, X_train, Y_train, 
                               lambda_theta=lambda_theta, epochs=theta_epochs, 
                               batch_size=batch_size, lr=lr_theta,
                               conv_threshold=conv_threshold, patience=patience)
        
        # Store theta model
        theta_models.append(theta_model)
    
    # Step 2: Cross-fit alpha models using pre-trained theta models
    print("Step 2: Cross-fitting alpha models...")
    for s in range(n_splits):
        print(f"Training alpha model for fold {s+1}/{n_splits}")
        
        # Get train and test indices for this split
        test_indices = split_indices[s]
        train_indices = np.concatenate([split_indices[j] for j in range(n_splits) if j != s])
        
        # Initialize alpha model - using linear model
        alpha_model = LinearAlpha(input_dim)  # Using LinearAlpha instead of AlphaNet
        optimizer = optim.Adam(alpha_model.parameters(), lr=lr_alpha)  
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=3, min_lr=1e-5)
        
        # Create a mapping from training sample to its original index
        train_to_original = {i: train_indices[i] for i in range(len(train_indices))}
        
        # Create dataset for training
        train_dataset = TensorDataset(torch.arange(len(train_indices)), 
                                     A_tensor[train_indices], 
                                     X_tensor[train_indices])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        alpha_model.g0.train()
        alpha_model.g1.train()
        
        # For early stopping
        best_loss = float('inf')
        best_model_state_g0 = None
        best_model_state_g1 = None
        patience_counter = 0
        prev_epoch_loss = float('inf')
        prev_lr = optimizer.param_groups[0]['lr']
        
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
                    theta_model = theta_models[fold]
                    theta_model.eval()
                    
                    # Calculate alpha loss using the corresponding theta model
                    with torch.no_grad():
                        # Get theta predictions
                        theta_axs = theta_model(fold_a, fold_x).squeeze()
                        
                        # Calculate sigmoid derivatives: sigmoid(x)*(1-sigmoid(x))
                        sigmoid_thetas = torch.sigmoid(theta_axs)
                        sigmoid_derivs = sigmoid_thetas * (1 - sigmoid_thetas)
                    
                    # Alpha for current (a,x) pairs
                    alpha_axs = alpha_model(fold_a, fold_x).squeeze()
                    
                    # Compute alpha(1,x) and alpha(0,x) for all x in this fold
                    alpha_1xs = alpha_model.g1(fold_x).squeeze()
                    alpha_0xs = alpha_model.g0(fold_x).squeeze()
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
                for param in alpha_model.parameters():
                    l2_reg += param.norm(2)
                
                batch_loss += lambda_alpha * l2_reg
                
                # Backward and optimize
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
            
            # Average loss for this epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
            
            # Update learning rate scheduler
            scheduler.step(avg_epoch_loss)
            
            # Track learning rate changes 
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                prev_lr = current_lr
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state_g0 = alpha_model.g0.state_dict().copy()
                best_model_state_g1 = alpha_model.g1.state_dict().copy()
            
            # Early stopping check
            if abs(prev_epoch_loss - avg_epoch_loss) < conv_threshold:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1} with loss change: {abs(prev_epoch_loss - avg_epoch_loss):.8f}")
                    break
            else:
                patience_counter = 0
                
            prev_epoch_loss = avg_epoch_loss
        
        # Restore best model
        if best_model_state_g0 is not None and best_model_state_g1 is not None:
            alpha_model.g0.load_state_dict(best_model_state_g0)
            alpha_model.g1.load_state_dict(best_model_state_g1)
        
        # Store alpha model
        alpha_models.append(alpha_model)
    
    # Step 3: Apply TMLE-inspired stabilization (optional)
    if stabilization:
        print("Applying TMLE-inspired stabilization...")
        
        # Use larger batch size for evaluation to reduce loop iterations
        batch_size_stab = min(5000, n)  # Use larger batch size
        n_batches_stab = int(np.ceil(n / batch_size_stab))
        
        numerator = 0
        denominator = 0
        
        for batch in range(n_batches_stab):
            start_idx = batch * batch_size_stab
            end_idx = min((batch + 1) * batch_size_stab, n)
            batch_indices = np.arange(start_idx, end_idx)
            
            batch_num = 0
            batch_denom = 0
            
            # Process samples in batch
            for i in batch_indices:
                fold = fold_mapping[i]
                
                theta_net = theta_models[fold]
                alpha_net = alpha_models[fold]
                
                with torch.no_grad():
                    # Calculate alpha difference
                    alpha_1x = alpha_net.g1(X_tensor[i:i+1]).item()
                    alpha_0x = alpha_net.g0(X_tensor[i:i+1]).item()
                    alpha_diff = alpha_1x - alpha_0x
                    
                    # Calculate denominator term
                    theta_ax = theta_net(A_tensor[i:i+1], X_tensor[i:i+1]).item()
                    sigmoid_theta = sigmoid(theta_ax)
                    sigmoid_deriv = sigmoid_theta * (1 - sigmoid_theta)
                    alpha_ax = alpha_net(A_tensor[i:i+1], X_tensor[i:i+1]).item()
                    denom_term = sigmoid_deriv * (alpha_ax ** 2)
                    
                    batch_num += alpha_diff
                    batch_denom += denom_term
            
            numerator += batch_num
            denominator += batch_denom
        
        # Calculate epsilon_n
        epsilon_n = numerator / denominator if denominator != 0 else 1.0
        print(f"  Stabilization factor epsilon_n = {epsilon_n:.4f}")
        
        # Store the epsilon_n value for scaling the alpha outputs
        stabilization_factor = epsilon_n
    else:
        stabilization_factor = 1.0  # No scaling if stabilization is disabled
    
    # Step 4: Compute one-step estimator
    print("Computing final estimate...")
    
    # Use larger batch size for evaluation
    batch_size_eval = min(5000, n)  # Use larger batch size
    n_batches = int(np.ceil(n / batch_size_eval))
    
    sum_term = 0
    
    for batch in range(n_batches):
        start_idx = batch * batch_size_eval
        end_idx = min((batch + 1) * batch_size_eval, n)
        batch_indices = np.arange(start_idx, end_idx)
        
        batch_sum = 0
        
        # Process samples in batch
        for i in batch_indices:
            fold = fold_mapping[i]
            
            # Get models trained without this sample
            theta_net = theta_models[fold]
            alpha_net = alpha_models[fold]
            
            # Get data for this sample
            x_i = X_tensor[i:i+1]
            a_i = A_tensor[i:i+1]
            y_i = Y_tensor[i:i+1].item()
            
            with torch.no_grad():
                # First term: theta(1,X) - theta(0,X)
                theta_1x = theta_net(torch.ones_like(a_i), x_i).item()
                theta_0x = theta_net(torch.zeros_like(a_i), x_i).item()
                theta_diff = theta_1x - theta_0x
                
                # Second term: (sigma(theta(A,X)) - Y) * alpha(A,X) * epsilon_n
                theta_ax = theta_net(a_i, x_i).item()
                sigma_theta = sigmoid(theta_ax)
                alpha_ax = alpha_net(a_i, x_i).item() * stabilization_factor  # Scale alpha output
                second_term = (sigma_theta - y_i) * alpha_ax
                
                # Combined term for this sample
                term_i = theta_diff - second_term
                batch_sum += term_i
        
        sum_term += batch_sum
    
    # Final estimate
    psi_hat = sum_term / n
    
    # Step 5: Compute standard error
    print("Computing standard error...")
    
    sum_squared_diff = 0
    
    for batch in range(n_batches):
        start_idx = batch * batch_size_eval
        end_idx = min((batch + 1) * batch_size_eval, n)
        batch_indices = np.arange(start_idx, end_idx)
        
        batch_sq_diff = 0
        
        for i in batch_indices:
            # Get the model fold for this sample
            fold = fold_mapping[i]
            
            # Get models trained without this sample
            theta_net = theta_models[fold]
            alpha_net = alpha_models[fold]
            
            # Get data for this sample
            x_i = X_tensor[i:i+1]
            a_i = A_tensor[i:i+1]
            y_i = Y_tensor[i:i+1].item()
            
            with torch.no_grad():
                # First term: theta(1,X) - theta(0,X)
                theta_1x = theta_net(torch.ones_like(a_i), x_i).item()
                theta_0x = theta_net(torch.zeros_like(a_i), x_i).item()
                theta_diff = theta_1x - theta_0x
                
                # Second term: (sigma(theta(A,X)) - Y) * alpha(A,X) * epsilon_n
                theta_ax = theta_net(a_i, x_i).item()
                sigma_theta = sigmoid(theta_ax)
                alpha_ax = alpha_net(a_i, x_i).item() * stabilization_factor  # Scale alpha output
                second_term = (sigma_theta - y_i) * alpha_ax
                
                # Combined term for this sample
                term_i = theta_diff - second_term
                
                # Squared difference for variance estimation
                squared_diff = (term_i - psi_hat) ** 2
                batch_sq_diff += squared_diff
        
        sum_squared_diff += batch_sq_diff
    
    # Estimated variance
    V_hat = sum_squared_diff / n
    
    # Standard error
    se_hat = np.sqrt(V_hat / n)
    
    # 95% confidence interval
    ci_lower = psi_hat - 1.96 * se_hat
    ci_upper = psi_hat + 1.96 * se_hat
    
    # Return results
    results = {
        'psi_hat': psi_hat,
        'se_hat': se_hat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'variance': V_hat
    }
    
    return results

# Process a single repetition
def process_single_rep(rep, size, true_psi, n_splits, lambda_theta, lambda_alpha, stabilization,
                       conv_threshold=1e-5, patience=5):
    """Process a single repetition of the simulation"""
    start_time = time.time()
    print(f"  Repetition {rep+1} for sample size {size}")
    
    # Generate data
    X, A, Y = generate_data(size)
    
    # Run autoDML with a unique seed for each repetition
    seed = 42 * size + rep  # Unique seed based on size and repetition
    res = autoDML(X, A, Y, n_splits=n_splits, lambda_theta=lambda_theta, lambda_alpha=lambda_alpha,
                 theta_epochs=200, alpha_epochs=200, stabilization=stabilization, seed=seed,
                 conv_threshold=conv_threshold, patience=patience)
    
    # Calculate metrics for this repetition
    bias = res['psi_hat'] - true_psi
    bias_se_ratio = bias / res['se_hat'] if res['se_hat'] > 0 else float('inf')
    covered = res['ci_lower'] <= true_psi <= res['ci_upper']
    ci_width = res['ci_upper'] - res['ci_lower']
    
    end_time = time.time()
    
    return {
        'psi_hat': res['psi_hat'],
        'se_hat': res['se_hat'],
        'ci_lower': res['ci_lower'],
        'ci_upper': res['ci_upper'],
        'bias': bias,
        'bias_se_ratio': bias_se_ratio,
        'covered': covered,
        'ci_width': ci_width,
        'time': end_time - start_time
    }

def run_simulation(sample_sizes=[1000, 5000, 10000], n_reps=100, n_splits=5, 
                  lambda_theta=0.001, lambda_alpha=0.001, stabilization=True,
                  conv_threshold=1e-5, patience=5, n_jobs=-1):
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
    stabilization : bool
        Whether to apply TMLE-inspired stabilization
    conv_threshold : float
        Convergence threshold for early stopping
    patience : int
        Number of epochs with small changes before early stopping
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
                      'biases': [], 'bias_se_ratios': [], 'coverages': [], 'ci_widths': []} 
               for size in sample_sizes}
    
    # Create directory for results
    os.makedirs('results', exist_ok=True)
    
    # Initialize results file
    with open('results/autodml_results.txt', 'w') as f:
        f.write("AutoDML Simulation Results\n")
        f.write("=========================\n\n")
        f.write(f"Number of repetitions: {n_reps}\n")
        f.write(f"Number of cross-fitting splits: {n_splits}\n")
        f.write(f"Regularization: lambda_theta={lambda_theta}, lambda_alpha={lambda_alpha}\n")
        f.write(f"TMLE-inspired stabilization: {stabilization}\n")
        f.write(f"Early stopping: threshold={conv_threshold}, patience={patience}\n")
        f.write(f"True target parameter: {true_psi}\n\n")
        f.write("Alpha values are scaled by epsilon_n\n\n")
    
    # Set up parallel processing
    n_cores = mp.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Using {n_cores} cores for parallel processing")
    
    # Run simulation for each sample size
    for size in sample_sizes:
        print(f"\n\nSimulating for sample size {size}")
        
        # Process repetitions in parallel
        rep_results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_rep)(
                rep=rep, 
                size=size, 
                true_psi=true_psi, 
                n_splits=n_splits, 
                lambda_theta=lambda_theta, 
                lambda_alpha=lambda_alpha, 
                stabilization=stabilization,
                conv_threshold=conv_threshold,
                patience=patience
            )
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
            
            # Write results to file
            with open('results/autodml_results.txt', 'a') as f:
                f.write(f"Sample size {size}, Rep {rep+1}: ")
                f.write(f"psi_hat={res['psi_hat']:.4f}, ")
                f.write(f"SE={res['se_hat']:.4f}, ")
                f.write(f"95% CI=[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}], ")
                f.write(f"bias={res['bias']:.4f}, ")
                f.write(f"bias/SE={res['bias_se_ratio']:.4f}, ")
                f.write(f"covered={res['covered']}, ")
                f.write(f"CI width={res['ci_width']:.4f}, ")
                f.write(f"time={res['time']:.2f}s\n")
        
        # Calculate and report aggregate metrics for this sample size
        psi_hats = np.array(results[size]['psi_hats'])
        ci_lowers = np.array(results[size]['ci_lowers'])
        ci_uppers = np.array(results[size]['ci_uppers'])
        ses = np.array(results[size]['ses'])
        biases = np.array(results[size]['biases'])
        bias_se_ratios = np.array(results[size]['bias_se_ratios'])
        
        mean_psi_hat = np.mean(psi_hats)
        mean_bias = np.mean(biases)
        mean_se = np.mean(ses)
        mean_bias_se_ratio = np.mean(bias_se_ratios)
        rmse = np.sqrt(np.mean((psi_hats - true_psi)**2))
        coverage = np.mean(results[size]['coverages'])
        mean_ci_width = np.mean(results[size]['ci_widths'])
        
        print(f"\nResults for sample size {size}:")
        print(f"  Mean estimate: {mean_psi_hat:.4f}")
        print(f"  Mean bias: {mean_bias:.4f}")
        print(f"  Mean standard error: {mean_se:.4f}")
        print(f"  Mean bias/SE ratio: {mean_bias_se_ratio:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Coverage: {coverage:.4f}")
        print(f"  Mean CI width: {mean_ci_width:.4f}")
        
        # Write aggregate results to file
        with open('results/autodml_results.txt', 'a') as f:
            f.write(f"\nAggregate results for sample size {size}:\n")
            f.write(f"  Mean estimate: {mean_psi_hat:.4f}\n")
            f.write(f"  Mean bias: {mean_bias:.4f}\n")
            f.write(f"  Mean standard error: {mean_se:.4f}\n")
            f.write(f"  Mean bias/SE ratio: {mean_bias_se_ratio:.4f}\n")
            f.write(f"  RMSE: {rmse:.4f}\n")
            f.write(f"  Coverage: {coverage:.4f}\n")
            f.write(f"  Mean CI width: {mean_ci_width:.4f}\n\n")
    
    # Create visualization of results
    plt.figure(figsize=(15, 15))
    
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
    plt.savefig('results/autodml_results.png')
    
    return results

if __name__ == "__main__":
    # Set hyperparameters
    n_reps = 100  # Number of simulation repetitions (increased from 5 to 100)
    n_splits = 5  # Number of cross-fitting splits
    lambda_theta = 0.001  # Regularization for theta
    lambda_alpha = 0.001  # Regularization for alpha
    stabilization = False  # Do not apply TMLE-inspired stabilization
    conv_threshold = 1e-5  # Convergence threshold for early stopping
    patience = 5  # Number of epochs with small change before stopping
    
    print("\nVerifying the theoretical target parameter...")
    # First verify the theoretical target parameter
    n_samples = 1000000
    X, _, _ = generate_data(n_samples)
    target_param = calculate_target_parameter(X)
    print(f"Target parameter with {n_samples} samples: {target_param:.4f}")
    print(f"Theoretical value: 1.1000")
    
    print("\nRunning autoDML simulation...")
    # Run the simulation with specified sample sizes
    results = run_simulation(
        sample_sizes=[1000, 5000, 10000, 30000, 50000],
        n_reps=n_reps,
        n_splits=n_splits,
        lambda_theta=lambda_theta,
        lambda_alpha=lambda_alpha,
        stabilization=stabilization,
        conv_threshold=conv_threshold,
        patience=patience,
        n_jobs=-1  # Use all available cores
    )
    
    print("\nSimulation complete. Results saved to 'results/autodml_results.txt' and 'results/autodml_results.png'.")