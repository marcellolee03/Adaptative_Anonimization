import numpy as np
import pandas as pd
import os
import sys
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import (SelectFromModel, SelectKBest, chi2,
                                       f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from file_utils import get_file, list_available_datasets
from ml import cross_validate_k_fold
from math_properties import calculate_math_properties, print_math_properties, export_math_properties


def feature_selection(X, y, method, k=None):
    if method == 'chi2':
        X = X.astype(np.float64)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        selector = SelectKBest(chi2, k=k)
        X_new = selector.fit_transform(X_scaled, y)
        selected_features_idx = selector.get_support(indices=True)
        return X_new, selected_features_idx
    elif method == 'extra_trees':
        X = X.astype(np.float64)
        
        model = ExtraTreesClassifier(n_estimators=100)
        model.fit(X, y)
        selector = SelectFromModel(model, prefit=True)
        X_new = selector.transform(X)
        selected_features_idx = selector.get_support(indices=True)
        return X_new, selected_features_idx
    else:
        raise ValueError(f"Feature selection method not supported: {method}")


def get_result(model, X, y, model_name, n_clusters, feature_method, k, noise_factor=0.01):
    bol = [True, False]
    results_columns = ['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score']
    results = pd.DataFrame(columns=results_columns)
    selected_features_all = []

    for i in range(0, 2):
        for j in range(0, 2):
            X_new, selected_features_idx = feature_selection(X, y, feature_method, k)
            selected_features_all.append({
                'anonymized_train': bol[i],
                'anonymized_test': bol[j],
                'model': model_name,
                'feature_method': feature_method,
                'num_features': k,
                'selected_features_idx': selected_features_idx.tolist()
            })
            cross_val_results = cross_validate_k_fold(X_new, y, bol[i], bol[j], model, model_name, n_clusters, noise_factor)
            new_df = pd.DataFrame([[
                model_name, 
                bol[i], 
                bol[j], 
                cross_val_results[2], 
                cross_val_results[3], 
                cross_val_results[4], 
                cross_val_results[5]
            ]], columns=results_columns)
            results = pd.concat([results, new_df], ignore_index=True)

    return results, selected_features_all

def experiment(X, y, feature_method, k, noise_factor=0.01):
    all_results = pd.DataFrame(columns=['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score', 'selected_features', 'feature_method', 'num_features'])
    models = [
        (KNeighborsClassifier(n_neighbors=5), 'KNN'),
        (RandomForestClassifier(n_estimators=100), 'Random Forest'),
        (GaussianNB(var_smoothing=1e-02), 'GaussianNB'),
        (MLPClassifier(
            hidden_layer_sizes=(100, 50),  
            activation='relu',             
            solver='adam',                 
            alpha=0.0001,                 
            learning_rate='adaptive',     
            learning_rate_init=0.001,      
            max_iter=500,                  
            early_stopping=True,           
            validation_fraction=0.1
        ), 'Multilayer Perceptron'),
        (AdaBoostClassifier(n_estimators=100, learning_rate=1.0), 'AdaBoost'),
        (LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='auto'), 'Logistic Regression')
    ]

    for model, model_name in models:
        results, selected_features = get_result(model, X, y, model_name, 3, feature_method, k, noise_factor)
        best_results = find_best_results(results, selected_features, feature_method, k)
        all_results = pd.concat([all_results, best_results], ignore_index=True)

    return all_results

def find_best_results(results, selected_features, feature_method, k):
    scenarios = [(True, True), (True, False), (False, True), (False, False)]
    best_results = []

    for model_name in results['model'].unique():
        for scenario in scenarios:
            anonymized_train, anonymized_test = scenario
            model_results = results[
                (results['model'] == model_name) &
                (results['anonymized_train'] == anonymized_train) &
                (results['anonymized_test'] == anonymized_test)
            ]
            if not model_results.empty:
                best_result = model_results.loc[model_results['accuracy'].idxmax()]
                selected_feature_info = [s for s in selected_features if 
                                          s['anonymized_train'] == anonymized_train and
                                          s['anonymized_test'] == anonymized_test and
                                          s['model'] == model_name and
                                          s['feature_method'] == feature_method and
                                          s['num_features'] == k]
                if selected_feature_info:
                    best_result.loc['selected_features'] = selected_feature_info[0]['selected_features_idx']
                best_result.loc['feature_method'] = feature_method
                best_result.loc['num_features'] = k
                best_results.append(best_result)

    return pd.DataFrame(best_results)

def Chi2(X, y, dataset_name, noise_factor=0.01):
    all_best_results = []
    
    # Get number of features from X shape
    num_features = X.shape[1]
    print(f"Running Chi2 with {num_features} features, noise factor: {noise_factor}")
    
    for i in range(2, num_features, 1):
        best_results = experiment(X, y, 'chi2', i, noise_factor)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    
    os.makedirs('results', exist_ok=True)
    
    # Include dataset name and noise factor in results filename
    filename = f'best_results_chi2_{dataset_name}_noise_{noise_factor:.2f}.csv'
    absolute_path = os.path.join(os.getcwd(), 'results', filename)
    final_best_results_df.to_csv(absolute_path, index=False)
    print(f"Chi2 results for {dataset_name} (noise: {noise_factor}) saved at: {absolute_path}")
    print(final_best_results_df.head())

def ExtraTree(X, y, dataset_name, noise_factor=0.01):
    all_best_results = []
    
    # Get number of features from X shape
    num_features = X.shape[1]
    print(f"Running ExtraTree with {num_features} features, noise factor: {noise_factor}")
    
    for i in range(2, num_features, 1):
        best_results = experiment(X, y, 'extra_trees', i, noise_factor)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    
    os.makedirs('results', exist_ok=True)
    
    # Include dataset name and noise factor in results filename
    filename = f'best_results_extra_trees_{dataset_name}_noise_{noise_factor:.2f}.csv'
    absolute_path = os.path.join(os.getcwd(), 'results', filename)
    final_best_results_df.to_csv(absolute_path, index=False)
    print(f"ExtraTree results for {dataset_name} (noise: {noise_factor}) saved at: {absolute_path}")

def math_properties_experiment(X, y, dataset_name, noise_factors=None):
    """
    Run experiment to calculate mathematical properties for different noise factors
    
    Args:
        X: Feature matrix
        y: Target vector
        dataset_name: Name of the dataset
        noise_factors: List of noise factors to test (default: [0.01, 0.05, 0.1, 0.5, 1.0])
    """
    if noise_factors is None:
        noise_factors = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    print("\n" + "="*80)
    print(f"RUNNING MATHEMATICAL PROPERTIES EXPERIMENT FOR {dataset_name.upper()}")
    print("="*80)
    
    all_properties = []
    
    for noise_factor in noise_factors:
        print(f"\nCalculating mathematical properties with noise factor = {noise_factor}")
        properties = calculate_math_properties(X, y, n_clusters=3, noise_factor=noise_factor)
        
        # Print and export results
        print_math_properties(properties)
        export_math_properties(properties, dataset_name, noise_factor)
        
        all_properties.append({
            'noise_factor': noise_factor,
            'properties': properties
        })
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON OF NOISE FACTORS")
    print("="*80)
    
    print("\nProperty | " + " | ".join([f"Noise={nf}" for nf in noise_factors]))
    print("-" * (80 + 10 * len(noise_factors)))
    
    key_metrics = [
        'anonymization_time', 
        'mean_difference',
        'std_difference',
        'covariance_similarity',
        'distance_correlation',
        'neighbor_preservation',
        'variance_preservation'
    ]
    
    for metric in key_metrics:
        values = [props['properties'][metric] for props in all_properties]
        print(f"{metric:20} | " + " | ".join([f"{val:.4f}" for val in values]))
    
    print("\n" + "="*80)
    print(f"Math properties experiment completed for {dataset_name}")
    print("="*80)
    
    # Export comparison summary
    summary_df = pd.DataFrame({
        'noise_factor': noise_factors,
        **{metric: [props['properties'][metric] for props in all_properties] for metric in key_metrics}
    })
    
    os.makedirs('results', exist_ok=True)
    summary_path = os.path.join(os.getcwd(), 'results', f'math_properties_summary_{dataset_name}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary of mathematical properties saved to: {summary_path}")

def main():
    np.random.seed(7)
    
    # Get dataset name, noise factor, and experiment type from command line
    dataset_name = None
    noise_factor = 0.01  # Default noise factor
    run_math_experiment = False
    run_ml_experiment = True  # Default to run ML experiment
    custom_noise_factors = None
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--noise='):
            try:
                noise_factor = float(arg.split('=')[1])
                print(f"Using custom noise factor: {noise_factor}")
            except (ValueError, IndexError):
                print(f"Invalid noise factor format. Using default: {noise_factor}")
        elif arg == '--math-only':
            run_math_experiment = True
            run_ml_experiment = False
        elif arg == '--math':
            run_math_experiment = True
        elif arg.startswith('--noise-levels='):
            try:
                # Parse comma-separated list of noise factors
                noise_str = arg.split('=')[1]
                custom_noise_factors = [float(nf) for nf in noise_str.split(',')]
                print(f"Using custom noise factors: {custom_noise_factors}")
            except (ValueError, IndexError):
                print("Invalid noise factors format. Using default levels.")
        elif i == 1 and not arg.startswith('--'):
            dataset_name = arg
    
    # Load the selected dataset
    dataset, label_column = get_file(dataset_name)
    
    # Extract dataset name for results file naming
    dataset_name = dataset_name or "cahousing"  # Default if none specified
    
    print(f"Total columns in dataset: {len(dataset.columns)}")
    print(f"Using '{label_column}' as target variable")
    
    # Extract feature and target
    y = np.array(dataset[label_column])
    dataset = dataset.drop(columns=[label_column])
    X = np.array(dataset)
    
    # Print dataset and feature shapes
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Number of unique classes: {len(np.unique(y))}")
    
    # Run selected experiments
    if run_math_experiment:
        if custom_noise_factors:
            math_properties_experiment(X, y, dataset_name, noise_factors=custom_noise_factors)
        else:
            math_properties_experiment(X, y, dataset_name)
            
    if run_ml_experiment:
        print(f"Anonymization noise factor for ML: {noise_factor}")
        Chi2(X, y, dataset_name, noise_factor)
        ExtraTree(X, y, dataset_name, noise_factor)


if __name__ == "__main__":
    # Display usage information if help flag is present
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['-h', '--help', 'help']:
        print("\nUsage: python main.py [dataset_name] [options]")
        print("\nOptions:")
        print("  dataset_name           Name of the dataset to use")
        print("  --noise=FACTOR         Noise factor for anonymization (default: 0.01)")
        print("  --math                 Run mathematical properties experiment")
        print("  --math-only            Run only mathematical properties experiment (skip ML)")
        print("  --noise-levels=N1,N2,N3 Comma-separated list of noise factors for math experiment")
        print("                         Example: --noise-levels=0.01,0.1,0.5,1.0")
        print("\nAvailable datasets:")
        list_available_datasets()
        sys.exit(0)
        
    main()