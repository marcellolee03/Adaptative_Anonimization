import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors

from anonimization import anonimization_clustering

def calculate_math_properties(X_orig, y_orig, n_clusters=3, noise_factor=0.01):
    """
    Calculate mathematical properties before and after anonymization
    
    Args:
        X_orig: Original feature matrix
        y_orig: Original target vector
        n_clusters: Number of clusters for anonymization
        noise_factor: Noise factor for anonymization
        
    Returns:
        Dictionary with properties before and after anonymization
    """
    results = {}
    
    # Ensure data is float64
    X_orig = X_orig.astype(np.float64)
    
    # Standardize for consistent comparison
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_orig)
    
    # Time the anonymization process
    start_time = time.time()
    X_anon, y_anon = anonimization_clustering(X_orig, y_orig, n_clusters, noise_factor=noise_factor)
    anon_time = time.time() - start_time
    results['anonymization_time'] = anon_time
    
    # Standardize anonymized data for comparison
    X_anon_std = scaler.transform(X_anon)
    
    # Calculate basic statistical properties
    results['orig_mean'] = np.mean(X_std, axis=0)
    results['anon_mean'] = np.mean(X_anon_std, axis=0)
    results['mean_difference'] = np.mean(np.abs(results['orig_mean'] - results['anon_mean']))
    
    results['orig_std'] = np.std(X_std, axis=0)
    results['anon_std'] = np.std(X_anon_std, axis=0)
    results['std_difference'] = np.mean(np.abs(results['orig_std'] - results['anon_std']))
    
    # Distribution shape metrics
    results['orig_skewness'] = skew(X_std, axis=0)
    results['anon_skewness'] = skew(X_anon_std, axis=0)
    results['skewness_difference'] = np.mean(np.abs(results['orig_skewness'] - results['anon_skewness']))
    
    results['orig_kurtosis'] = kurtosis(X_std, axis=0)
    results['anon_kurtosis'] = kurtosis(X_anon_std, axis=0)
    results['kurtosis_difference'] = np.mean(np.abs(results['orig_kurtosis'] - results['anon_kurtosis']))
    
    # Covariance structure
    cov_orig = EmpiricalCovariance().fit(X_std)
    cov_anon = EmpiricalCovariance().fit(X_anon_std)
    results['orig_covariance_norm'] = np.linalg.norm(cov_orig.covariance_)
    results['anon_covariance_norm'] = np.linalg.norm(cov_anon.covariance_)
    results['covariance_difference'] = np.linalg.norm(cov_orig.covariance_ - cov_anon.covariance_)
    results['covariance_similarity'] = np.trace(cov_orig.covariance_ @ cov_anon.covariance_) / (
        np.linalg.norm(cov_orig.covariance_) * np.linalg.norm(cov_anon.covariance_))
    
    # Distance preservation
    # Calculate pairwise distances in original and anonymized data for a sample
    max_samples = min(1000, X_std.shape[0])  # Limit sample size for memory constraints
    sample_indices = np.random.choice(X_std.shape[0], max_samples, replace=False)
    
    dist_orig = euclidean_distances(X_std[sample_indices])
    dist_anon = euclidean_distances(X_anon_std[sample_indices])
    
    # Flatten the distance matrices
    dist_orig_flat = dist_orig[np.triu_indices(dist_orig.shape[0], k=1)]
    dist_anon_flat = dist_anon[np.triu_indices(dist_anon.shape[0], k=1)]
    
    # Calculate correlation between original and anonymized distances
    results['distance_correlation'] = np.corrcoef(dist_orig_flat, dist_anon_flat)[0, 1]
    
    # Check nearest neighbor preservation
    k = min(5, max_samples - 1)  # Number of neighbors to check
    nn_orig = NearestNeighbors(n_neighbors=k+1).fit(X_std[sample_indices])
    nn_anon = NearestNeighbors(n_neighbors=k+1).fit(X_anon_std[sample_indices])
    
    # Get indices of nearest neighbors
    neighbors_orig = nn_orig.kneighbors(X_std[sample_indices], return_distance=False)[:,1:]  # Skip self
    neighbors_anon = nn_anon.kneighbors(X_anon_std[sample_indices], return_distance=False)[:,1:]  # Skip self
    
    # Calculate neighbor preservation rate
    neighbor_preservation = 0
    for i in range(max_samples):
        orig_neighbors = set(neighbors_orig[i])
        anon_neighbors = set(neighbors_anon[i])
        overlap = len(orig_neighbors.intersection(anon_neighbors))
        neighbor_preservation += overlap / k
    
    results['neighbor_preservation'] = neighbor_preservation / max_samples
    
    # PCA variance preservation
    pca_orig = PCA().fit(X_std)
    pca_anon = PCA().fit(X_anon_std)
    
    results['orig_explained_variance'] = pca_orig.explained_variance_ratio_
    results['anon_explained_variance'] = pca_anon.explained_variance_ratio_
    
    # Get the minimum length
    min_length = min(len(results['orig_explained_variance']), len(results['anon_explained_variance']))
    results['variance_preservation'] = np.sum(results['orig_explained_variance'][:min_length] * 
                                              results['anon_explained_variance'][:min_length]) / np.sum(
                                              results['orig_explained_variance'][:min_length])
    
    # Class separation metric
    # Calculate mean distance between classes
    unique_classes = np.unique(y_orig)
    orig_class_means = np.array([np.mean(X_std[y_orig == cls], axis=0) for cls in unique_classes])
    anon_class_means = np.array([np.mean(X_anon_std[y_anon == cls], axis=0) for cls in unique_classes])
    
    orig_between_class_dist = euclidean_distances(orig_class_means)
    anon_between_class_dist = euclidean_distances(anon_class_means)
    
    # Flatten and correlate
    orig_between_flat = orig_between_class_dist[np.triu_indices(orig_between_class_dist.shape[0], k=1)]
    anon_between_flat = anon_between_class_dist[np.triu_indices(anon_between_class_dist.shape[0], k=1)]
    
    if len(orig_between_flat) > 0:  # Only calculate if we have multiple classes
        results['class_separation_correlation'] = np.corrcoef(orig_between_flat, anon_between_flat)[0, 1]
    else:
        results['class_separation_correlation'] = np.nan
    
    return results


def print_math_properties(properties):
    """
    Print mathematical properties in a formatted way
    
    Args:
        properties: Dictionary with mathematical properties
    """
    print("\n" + "="*80)
    print("MATHEMATICAL PROPERTIES REPORT")
    print("="*80)
    
    print(f"\nAnonymization time: {properties['anonymization_time']:.4f} seconds")
    
    print("\nBASIC STATISTICAL PROPERTIES")
    print("-"*50)
    print(f"Mean difference: {properties['mean_difference']:.4f}")
    print(f"Standard deviation difference: {properties['std_difference']:.4f}")
    print(f"Skewness difference: {properties['skewness_difference']:.4f}")
    print(f"Kurtosis difference: {properties['kurtosis_difference']:.4f}")
    
    print("\nSTRUCTURAL PROPERTIES")
    print("-"*50)
    print(f"Covariance matrix norm (original): {properties['orig_covariance_norm']:.4f}")
    print(f"Covariance matrix norm (anonymized): {properties['anon_covariance_norm']:.4f}")
    print(f"Covariance difference: {properties['covariance_difference']:.4f}")
    print(f"Covariance similarity (cosine): {properties['covariance_similarity']:.4f}")
    
    print("\nDISTANCE PRESERVATION")
    print("-"*50)
    print(f"Pairwise distance correlation: {properties['distance_correlation']:.4f}")
    print(f"Nearest neighbor preservation: {properties['neighbor_preservation']:.4f}")
    
    print("\nPCA VARIANCE PRESERVATION")
    print("-"*50)
    print(f"Variance preservation ratio: {properties['variance_preservation']:.4f}")
    
    # Compare top 3 principal components' explained variance
    n_components = min(3, len(properties['orig_explained_variance']), len(properties['anon_explained_variance']))
    print(f"\nTop {n_components} principal components explained variance:")
    for i in range(n_components):
        print(f"  PC{i+1}: Original={properties['orig_explained_variance'][i]:.4f}, "
              f"Anonymized={properties['anon_explained_variance'][i]:.4f}")
    
    print("\nCLASS SEPARATION")
    print("-"*50)
    if not np.isnan(properties['class_separation_correlation']):
        print(f"Class separation correlation: {properties['class_separation_correlation']:.4f}")
    else:
        print("Class separation correlation: N/A (only one class)")
    
    print("\n" + "="*80)


def export_math_properties(properties, dataset_name, noise_factor, output_dir='results'):
    """
    Export mathematical properties to CSV
    
    Args:
        properties: Dictionary with mathematical properties
        dataset_name: Name of the dataset
        noise_factor: Noise factor used
        output_dir: Directory to save results
    """
    import os
    import pandas as pd
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV
    data = {
        'dataset': dataset_name,
        'noise_factor': noise_factor,
        'anonymization_time': properties['anonymization_time'],
        'mean_difference': properties['mean_difference'],
        'std_difference': properties['std_difference'],
        'skewness_difference': properties['skewness_difference'],
        'kurtosis_difference': properties['kurtosis_difference'],
        'covariance_difference': properties['covariance_difference'],
        'covariance_similarity': properties['covariance_similarity'],
        'distance_correlation': properties['distance_correlation'],
        'neighbor_preservation': properties['neighbor_preservation'],
        'variance_preservation': properties['variance_preservation'],
        'class_separation_correlation': properties['class_separation_correlation']
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Save to CSV
    filename = f'math_properties_{dataset_name}_noise_{noise_factor:.2f}.csv'
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"Mathematical properties saved to: {filepath}")