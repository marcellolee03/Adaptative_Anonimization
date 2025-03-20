import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2


def calculate_feature_importance(X, y):
    """Calculate feature importance using ExtraTreesClassifier."""
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model.feature_importances_


def add_laplace_noise(data, feature_importances, epsilon=1.0, scale_factor=0.1):
    """Add Laplace noise with differential privacy guarantees.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The data to add noise to
    feature_importances : numpy.ndarray
        Importance scores for each feature
    epsilon : float
        Privacy parameter - smaller values provide more privacy
    scale_factor : float
        Factor to control noise level
    
    Returns:
    --------
    numpy.ndarray
        Data with Laplace noise added
    """
    # Normalize feature importances to range [0, 1]
    if feature_importances is not None:
        norm_importances = feature_importances / np.max(feature_importances)
        # Invert so less important features get more noise
        sensitivity = 1 - norm_importances
    else:
        # If no feature importances provided, use uniform sensitivity
        sensitivity = np.ones(data.shape[1])
    
    # Add noise proportional to sensitivity and scaled by epsilon
    noisy_data = data.copy()
    for i in range(data.shape[1]):
        # Scale determines noise magnitude (smaller epsilon = more noise)
        scale = scale_factor * sensitivity[i] / epsilon
        noise = np.random.laplace(0, scale, size=data.shape[0])
        noisy_data[:, i] = data[:, i] + noise
        
    return noisy_data


def anonimization(data, feature_importances=None, epsilon=1.0):
    """Anonymize data using PCA transformation and differential privacy."""
    # Calculate the mean of each column
    mean = np.array(np.mean(data, axis=0).T)

    # Center data
    data_centered = data - mean

    # Calculate the covariance matrix
    cov_matrix = np.cov(data_centered, rowvar=False)
   
    # Calculate the eigenvalues and eigenvectors
    evals, evecs = la.eigh(cov_matrix)

    # Sort them
    idx = np.argsort(evals)[::-1]

    # Each column of this matrix is an eigenvector
    evecs = evecs[:, idx]
    evals = evals[idx]

    # Calculate the transformed data
    data_transformed = np.dot(evecs.T, data_centered.T).T

    # Apply differential privacy with Laplace noise
    data_transformed_noisy = add_laplace_noise(
        data_transformed, 
        feature_importances, 
        epsilon=epsilon
    )

    # Go back to the original dimension
    data_original_dimension = np.dot(data_transformed_noisy, evecs.T) 
    data_original_dimension += mean

    return data_original_dimension


def find_clusters(X, k):   
    Kmean = KMeans(n_clusters=k, random_state=42)
    Kmean.fit(X)
    return Kmean.labels_


def anonimization_clustering(data, y, k, epsilon=1.0):
    """Anonymize data clustered into k groups with differential privacy."""
    # Generate K data clusters
    clusters = find_clusters(data, k)

    # Calculate feature importance for noise scaling
    feature_importances = calculate_feature_importance(data, y)

    # Bucketize the index of each cluster
    indices = dict()
    for i in range(len(clusters)):
        if clusters[i] not in indices.keys():
            indices[clusters[i]] = []    
        indices[clusters[i]].append(i)

    data_anonymized, y_in_new_order = None, None

    # Anonymize each cluster individually
    for k in indices.keys():
        if data_anonymized is None and y_in_new_order is None:
            data_anonymized = anonimization(
                data[indices[k]], 
                feature_importances,
                epsilon
            )
            y_in_new_order = y[indices[k]]
        else:
            data_anonymized = np.concatenate(
                (data_anonymized, anonimization(data[indices[k]], feature_importances, epsilon)), 
                axis=0
            )
            y_in_new_order = np.concatenate((y_in_new_order, y[indices[k]]), axis=0)

    return data_anonymized, y_in_new_order


def feature_selection(X, y, method, k=None):
    """Select features using specified method."""
    if method == 'chi2':
        selector = SelectKBest(chi2, k=k)
    elif method == 'extra_trees':
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        selector = SelectFromModel(model, max_features=k, prefit=True)
    else:
        raise ValueError(f"Método de seleção de características não suportado: {method}")

    X_new = selector.fit_transform(X, y)
    selected_features_idx = selector.get_support(indices=True)
    return X_new, selected_features_idx