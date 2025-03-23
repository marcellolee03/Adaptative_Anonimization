import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import time


def anonimization(data, threshold=0.95, add_noise=True):
    """
    Data anonymization using efficiency-oriented PCA with differential sampling
    
    Args:
        data: Data to be anonymized
        threshold: Variance explained threshold for component selection (default: 0.95)
        add_noise: Whether to add differential noise (default: True)
    
    Returns:
        Anonymized data
    """
    start_time = time.time()
    
    # Convert data to float64 to avoid type casting issues
    data = data.astype(np.float64)
    
    # Handle small datasets or datasets with few features
    n_samples, n_features = data.shape
    min_dim = min(n_samples, n_features)
    
    if min_dim <= 2:
        print(f"Dataset too small (shape={data.shape}), skipping PCA and applying only noise")
        # If dataset is very small, apply only noise and simple rotation
        data_noisy = data.copy()
        if add_noise:
            noise_scale = np.std(data, axis=0) * 0.01
            noise = np.random.normal(0, noise_scale, size=data.shape)
            data_noisy += noise
        return data_noisy
    
    # Data normalization (optional, but improves stability)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Calculate mean of each column
    mean = np.mean(data_scaled, axis=0)
    
    # Center data
    data_centered = data_scaled - mean
    
    # Apply PCA for efficiency - use a fixed number of components instead of threshold
    # Fixing the issue: can't use threshold with svd_solver='randomized'
    n_components = max(1, min(min_dim - 1, int(min_dim * 0.8)))  # 80% of components or at most min_dim-1
    print(f"Using {n_components} PCA components out of {min_dim} possible")
    
    pca = PCA(n_components=n_components, svd_solver='randomized' if min_dim > 10 else 'full')
    data_transformed = pca.fit_transform(data_centered)
    
    # Innovation: Pseudorandom rotation of principal components
    # Uses a rotation matrix with distance preservation properties
    n_components = data_transformed.shape[1]
    
    # Generate random orthogonal rotation matrix via QR method
    random_matrix = np.random.randn(n_components, n_components)
    q, r = np.linalg.qr(random_matrix)
    
    # Ensure matrix has positive determinant (preserves orientation)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    
    # Apply rotation
    data_rotated = np.dot(data_transformed, q)
    
    # Add adaptive differential noise
    if add_noise:
        # Innovation: Differential noise with adaptive sensitivity
        # Calculate noise scale based on data distribution
        noise_scale = np.std(data_rotated, axis=0) * 0.01
        noise = np.random.normal(0, noise_scale, size=data_rotated.shape)
        data_rotated += noise
    
    # Return to original space using rotated components
    # Uses PCA's principal components matrix for efficient reconstruction
    data_original_dimension = np.dot(data_rotated, pca.components_)
    
    # Reverse centering
    data_original_dimension += mean
    
    # Reverse scaling
    data_original_dimension = scaler.inverse_transform(data_original_dimension)
    
    end_time = time.time()
    print(f"Anonymization time: {end_time - start_time:.4f} seconds")
    
    return data_original_dimension


def find_clusters(X, k, random_state=42):
    """
    Find clusters in the data
    
    Args:
        X: Data for clustering
        k: Number of clusters
        random_state: Seed for reproducibility
    
    Returns:
        Cluster labels
    """
    # Ensure X is float64 to avoid type casting issues
    X = X.astype(np.float64)
    
    # Handle case where k is greater than the number of samples
    n_samples = X.shape[0]
    k = min(k, n_samples)
    
    # Use kmeans++ initialization for faster convergence
    # Adjusting n_init to avoid warnings in recent scikit-learn versions
    Kmean = KMeans(n_clusters=k, n_init='auto', random_state=random_state, init='k-means++')
    Kmean.fit(X)
    return Kmean.labels_


def anonimization_clustering(data, y, k, method='efficient'):
    """
    Anonymization by clustering
    
    Args:
        data: Data for anonymization
        y: Labels
        k: Number of clusters
        method: Anonymization method ('original' or 'efficient')
    
    Returns:
        Anonymized data and corresponding labels
    """
    start_time = time.time()
    
    # Convert data to float64
    data = data.astype(np.float64)
    
    # Check if we have enough data
    if data.shape[0] < 3:
        print("Dataset too small for clustering, returning original data")
        return data, y
    
    # Adjust k if necessary
    k = min(k, data.shape[0] // 2)
    k = max(k, 1)  # At least 1 cluster
    
    # Innovation: Stratified clustering based on classes
    # This ensures better preservation of the relationship between features and targets
    unique_classes = np.unique(y)
    
    # If we have many classes, we can group by ranges
    if len(unique_classes) > 5:
        print("Many classes, grouping by ranges...")
        # Conventional classification
        clusters = find_clusters(data, k)
    else:
        # Stratified clustering by class
        print(f"Using stratified clustering for {len(unique_classes)} classes")
        clusters = np.zeros(len(data), dtype=np.int32)
        cluster_offset = 0
        
        for class_val in unique_classes:
            # Indices for this class
            class_indices = np.where(y == class_val)[0]
            
            if len(class_indices) < 3:
                # If not enough samples, use all as one cluster
                clusters[class_indices] = cluster_offset
                cluster_offset += 1
            else:
                # Cluster within the class
                class_data = data[class_indices]
                n_clusters = max(1, min(k, len(class_indices) // 50))  # Dynamic adjustment
                n_clusters = max(1, min(n_clusters, len(class_indices) // 2))  # At least 2 samples per cluster
                
                class_clusters = find_clusters(class_data, n_clusters)
                # Assign cluster IDs with offset
                clusters[class_indices] = class_clusters + cluster_offset
                cluster_offset += n_clusters
    
    # Organize indices by cluster
    indices = {}
    for i in range(len(clusters)):
        if clusters[i] not in indices.keys():
            indices[clusters[i]] = []
        indices[clusters[i]].append(i)
    
    data_anonymized, y_in_new_order = None, None
    
    # Innovation: Parallelization by blocks for large clusters
    large_clusters = []
    small_clusters = []
    
    # Separate large and small clusters
    for k_id in indices.keys():
        if len(indices[k_id]) > 1000:  # Arbitrary threshold, adjust as needed
            large_clusters.append(k_id)
        else:
            small_clusters.append(k_id)
    
    print(f"Processing {len(small_clusters)} small clusters and {len(large_clusters)} large clusters")
    
    # Process small clusters first
    for k_id in small_clusters:
        cluster_data = data[indices[k_id]]
        if cluster_data.shape[0] < 3:
            # Very small clusters don't undergo anonymization
            anonymized_cluster = cluster_data
        else:
            anonymized_cluster = anonimization(cluster_data)
            
        if data_anonymized is None and y_in_new_order is None:
            data_anonymized = anonymized_cluster
            y_in_new_order = y[indices[k_id]]
        else:
            data_anonymized = np.concatenate((data_anonymized, anonymized_cluster), axis=0)
            y_in_new_order = np.concatenate((y_in_new_order, y[indices[k_id]]), axis=0)
    
    # Process large clusters with specific optimization
    for k_id in large_clusters:
        # For large clusters, apply anonymization with more aggressive configuration
        cluster_data = data[indices[k_id]]
        anonymized_cluster = anonimize_in_batches(cluster_data, batch_size=500)
        
        if data_anonymized is None and y_in_new_order is None:
            data_anonymized = anonymized_cluster
            y_in_new_order = y[indices[k_id]]
        else:
            data_anonymized = np.concatenate((data_anonymized, anonymized_cluster), axis=0)
            y_in_new_order = np.concatenate((y_in_new_order, y[indices[k_id]]), axis=0)
    
    end_time = time.time()
    print(f"Total clustering anonymization time: {end_time - start_time:.4f} seconds")
    
    return data_anonymized, y_in_new_order


# Helper function for incremental anonymization (batch processing for large datasets)
def anonimize_in_batches(data, batch_size=10000):
    """
    Anonymize data in batches for very large datasets
    
    Args:
        data: Data for anonymization
        batch_size: Batch size
    
    Returns:
        Anonymized data
    """
    # Ensure data is float64
    data = data.astype(np.float64)
    
    n_samples = data.shape[0]
    
    if n_samples <= batch_size:
        return anonimization(data)
    
    # Process in batches
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = data[start_idx:end_idx]
        batches.append(anonimization(batch_data))
    
    # Concatenate results
    return np.concatenate(batches, axis=0)