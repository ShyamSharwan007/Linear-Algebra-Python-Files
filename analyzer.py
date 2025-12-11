import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def perform_analysis(df_zscore, scenario_name="Scenario"):
    """
    Performs PCA and SVD analysis on the Z-Score normalized data.
    
    Args:
        df_zscore: Z-Score normalized DataFrame
        scenario_name: Name of the scenario for plot titles
        
    Returns:
        pca_results: Dictionary containing PCA components and explained variance
        svd_results: Dictionary containing U, S, V matrices
    """
    # --- PCA Analysis ---
    pca = PCA(n_components=2) # Reduce to 2D for visualization
    principal_components = pca.fit_transform(df_zscore)
    
    explained_variance = pca.explained_variance_ratio_
    
    # Create Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', s=100)
    
    for i, txt in enumerate(df_zscore.index):
        plt.annotate(txt, (principal_components[i, 0], principal_components[i, 1]), xytext=(5, 5), textcoords='offset points')
        
    plt.title(f'PCA - 3PL Performance Clustering ({scenario_name})')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} Variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} Variance)')
    plt.grid(True)
    plt.savefig(f'{scenario_name}_pca_plot.png')
    plt.close()
    
    # --- SVD Analysis ---
    # X = U * S * Vt
    U, S, Vt = np.linalg.svd(df_zscore, full_matrices=False)
    
    # Low-Rank Approximation (Denoising) - Slide 9 Step 7
    # Reconstruct using top k singular values (e.g., k=2) to capture main trends and remove noise
    k = 2
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])
    
    # X_reconstructed = U[:, :k] * S_k * Vt[:k, :]
    X_reconstructed = np.dot(np.dot(U[:, :k], S_k), Vt[:k, :])
    df_reconstructed = pd.DataFrame(X_reconstructed, index=df_zscore.index, columns=df_zscore.columns)
    
    return {
        'pca_components': principal_components,
        'explained_variance': explained_variance,
        'U': U,
        'S': S,
        'Vt': Vt,
        'reconstructed_data': df_reconstructed
    }
