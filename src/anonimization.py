import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import time


def anonimization(data, threshold=0.95, add_noise=True, noise_factor=0.01):
    """
    Data anonymization using efficiency-oriented PCA with differential sampling
    
    Args:
        data: Data to be anonymized
        threshold: Variance explained threshold for component selection (default: 0.95)
        add_noise: Whether to add differential noise (default: True)
        noise_factor: Factor to control noise magnitude (default: 0.01, higher = more noise)
    
    Returns:
        Anonymized data
    """
    start_time = time.time()
    
    data = data.astype(np.float64)
    
    n_samples, n_features = data.shape
    min_dim = min(n_samples, n_features)
    
    if min_dim <= 2:
        print(f"Dataset too small (shape={data.shape}), skipping PCA and applying only noise")
        data_noisy = data.copy()
        if add_noise:
            noise_scale = np.std(data, axis=0) * noise_factor
            noise = np.random.normal(0, noise_scale, size=data.shape)
            data_noisy += noise
        return data_noisy
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    mean = np.mean(data_scaled, axis=0)
    
    data_centered = data_scaled - mean
    
    n_components = max(1, min(min_dim - 1, int(min_dim * 0.8)))
    print(f"Using {n_components} PCA components out of {min_dim} possible")
    
    pca = PCA(n_components=n_components, svd_solver='randomized' if min_dim > 10 else 'full')
    data_transformed = pca.fit_transform(data_centered)
    
    n_components = data_transformed.shape[1]
    
    random_matrix = np.random.randn(n_components, n_components)
    q, r = np.linalg.qr(random_matrix)
    
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    
    data_rotated = np.dot(data_transformed, q)
    
    if add_noise:
        # Modified noise scale with adjustable factor
        noise_scale = np.std(data_rotated, axis=0) * noise_factor
        noise = np.random.normal(0, noise_scale, size=data_rotated.shape)
        data_rotated += noise
    
    data_original_dimension = np.dot(data_rotated, pca.components_)
    
    data_original_dimension += mean
    
    data_original_dimension = scaler.inverse_transform(data_original_dimension)
    
    end_time = time.time()
    print(f"Anonymization time: {end_time - start_time:.4f} seconds")
    
    return data_original_dimension


def anonimization_clustering(data, y, k, method='efficient', noise_factor=0.01):
    """
    Anonymization by clustering
    
    Args:
        data: Data for anonymization
        y: Labels
        k: Number of clusters
        method: Anonymization method ('original' or 'efficient')
        noise_factor: Factor to control noise magnitude (default: 0.01, higher = more noise)
    
    Returns:
        Anonymized data and corresponding labels
    """
    start_time = time.time()
    
    data = data.astype(np.float64)
    
    if data.shape[0] < 3:
        print("Dataset too small for clustering, returning original data")
        return data, y
    
    k = min(k, data.shape[0] // 2)
    k = max(k, 1)
    
    unique_classes = np.unique(y)
    
    if len(unique_classes) > 5:
        print("Many classes, grouping by ranges...")
        clusters = find_clusters(data, k)
    else:
        print(f"Using stratified clustering for {len(unique_classes)} classes")
        clusters = np.zeros(len(data), dtype=np.int32)
        cluster_offset = 0
        
        for class_val in unique_classes:
            class_indices = np.where(y == class_val)[0]
            
            if len(class_indices) < 3:
                clusters[class_indices] = cluster_offset
                cluster_offset += 1
            else:
                class_data = data[class_indices]
                n_clusters = max(1, min(k, len(class_indices) // 50))
                n_clusters = max(1, min(n_clusters, len(class_indices) // 2))
                
                class_clusters = find_clusters(class_data, n_clusters)
                clusters[class_indices] = class_clusters + cluster_offset
                cluster_offset += n_clusters
    
    indices = {}
    for i in range(len(clusters)):
        if clusters[i] not in indices.keys():
            indices[clusters[i]] = []
        indices[clusters[i]].append(i)
    
    data_anonymized, y_in_new_order = None, None
    
    large_clusters = []
    small_clusters = []
    
    for k_id in indices.keys():
        if len(indices[k_id]) > 1000:
            large_clusters.append(k_id)
        else:
            small_clusters.append(k_id)
    
    print(f"Processing {len(small_clusters)} small clusters and {len(large_clusters)} large clusters")
    
    for k_id in small_clusters:
        cluster_data = data[indices[k_id]]
        if cluster_data.shape[0] < 3:
            anonymized_cluster = cluster_data
        else:
            anonymized_cluster = anonimization(cluster_data, noise_factor=noise_factor)
            
        if data_anonymized is None and y_in_new_order is None:
            data_anonymized = anonymized_cluster
            y_in_new_order = y[indices[k_id]]
        else:
            data_anonymized = np.concatenate((data_anonymized, anonymized_cluster), axis=0)
            y_in_new_order = np.concatenate((y_in_new_order, y[indices[k_id]]), axis=0)
    
    for k_id in large_clusters:
        cluster_data = data[indices[k_id]]
        anonymized_cluster = anonimize_in_batches(cluster_data, batch_size=500, noise_factor=noise_factor)
        
        if data_anonymized is None and y_in_new_order is None:
            data_anonymized = anonymized_cluster
            y_in_new_order = y[indices[k_id]]
        else:
            data_anonymized = np.concatenate((data_anonymized, anonymized_cluster), axis=0)
            y_in_new_order = np.concatenate((y_in_new_order, y[indices[k_id]]), axis=0)
    
    end_time = time.time()
    print(f"Total clustering anonymization time: {end_time - start_time:.4f} seconds")
    
    return data_anonymized, y_in_new_order


def find_clusters(X, k, random_state=42):
    """
    Find clusters in the data
    """
    X = X.astype(np.float64)
    
    n_samples = X.shape[0]
    k = min(k, n_samples)
    
    Kmean = KMeans(n_clusters=k, n_init='auto', random_state=random_state, init='k-means++')
    Kmean.fit(X)
    return Kmean.labels_


def anonimize_in_batches(data, batch_size=10000, noise_factor=0.01):
    """
    Anonymize data in batches for very large datasets
    
    Args:
        data: Data for anonymization
        batch_size: Batch size
        noise_factor: Factor to control noise magnitude (default: 0.01, higher = more noise)
    
    Returns:
        Anonymized data
    """
    data = data.astype(np.float64)
    
    n_samples = data.shape[0]
    
    if n_samples <= batch_size:
        return anonimization(data, noise_factor=noise_factor)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = data[start_idx:end_idx]
        batches.append(anonimization(batch_data, noise_factor=noise_factor))
    
    return np.concatenate(batches, axis=0)