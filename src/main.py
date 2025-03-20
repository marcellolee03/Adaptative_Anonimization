import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from anonimization import feature_selection
from file_utils import get_file
from ml import cross_validate_k_fold


# Function to get the results
def get_result(model, X, y, model_name, n_clusters, feature_method, k, epsilon=1.0):
    bol = [True, False]
    results_columns = ['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'inference_time']
    results = pd.DataFrame(columns=results_columns)
    selected_features_all = []

    for i in range(0, 2):
        for j in range(0, 2):
            # Feature Selection
            X_new, selected_features_idx = feature_selection(X, y, feature_method, k)
            selected_features_all.append({
                'anonymized_train': bol[i],
                'anonymized_test': bol[j],
                'model': model_name,
                'feature_method': feature_method,
                'num_features': k,
                'selected_features_idx': selected_features_idx.tolist()
            })
            cross_val_results = cross_validate_k_fold(X_new, y, bol[i], bol[j], model, model_name, n_clusters, epsilon)
            new_df = pd.DataFrame([[
                model_name, 
                bol[i], 
                bol[j], 
                cross_val_results[2], 
                cross_val_results[3], 
                cross_val_results[4], 
                cross_val_results[5],
                cross_val_results[6],  # training_time
                cross_val_results[7]   # inference_time
            ]], columns=results_columns)
            results = pd.concat([results, new_df], ignore_index=True)

    return results, selected_features_all


# Function where the experiment is made
def experiment(X, y, feature_method, k, epsilon=1.0):
    all_results = pd.DataFrame(columns=['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'inference_time', 'selected_features', 'feature_method', 'num_features', 'epsilon'])
    models = [
        (KNeighborsClassifier(n_neighbors=5), 'KNN'),
        (AdaBoostClassifier(n_estimators=100), 'AdaBoost'),
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
        (LogisticRegression(
            C=1.0, 
            max_iter=500, 
            solver='lbfgs', 
            multi_class='auto', 
            class_weight='balanced',
            random_state=42
        ), 'Logistic Regression')
    ]

    for model, model_name in models:
        results, selected_features = get_result(model, X, y, model_name, 3, feature_method, k, epsilon)
        # Find the best result for each scenario
        best_results = find_best_results(results, selected_features, feature_method, k, epsilon)
        all_results = pd.concat([all_results, best_results], ignore_index=True)

    return all_results


def find_best_results(results, selected_features, feature_method, k, epsilon):
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
                best_result.loc['epsilon'] = epsilon
                best_results.append(best_result)

    return pd.DataFrame(best_results)


def Chi2(X, y):
    all_best_results = []
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Diferentes níveis de privacidade diferencial

    for epsilon in epsilon_values:
        # for i in range(2, 10, 1):
        #     best_results = experiment(X, y, 'chi2', i, epsilon)
        #     all_best_results.append(best_results)

        for i in range(2, 48, 5):
            best_results = experiment(X, y, 'chi2', i, epsilon)
            all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    final_best_results_df.to_csv('results/best_results_chi2_dp.csv', index=False)
    print(final_best_results_df.head())


def ExtraTrees(X, y):
    all_best_results = []
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Diferentes níveis de privacidade diferencial

    for epsilon in epsilon_values:
        for i in range(2, 48, 5):
            best_results = experiment(X, y, 'extra_trees', i, epsilon)
            all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    final_best_results_df.to_csv('results/best_results_extra_trees_dp.csv', index=False)
    print(final_best_results_df.head())


def main():
    # Set the random seed 
    np.random.seed(7) 

    # Read the dataset
    dataset = get_file()
    # print(len(dataset.columns))
    # print(dataset.columns)

    # Extract feature and label
    if 'Label' in dataset.columns:
        y = np.array(dataset['Label'])
        dataset = dataset.drop('Label', axis=1)
        X = np.array(dataset)
        
        # Choose the feature selection method
        # print("Executando Chi2...")
        # Chi2(X, y)
        
        print("Executando ExtraTrees...")
        ExtraTrees(X, y)
    else:
        print("Erro: Coluna 'Label' não encontrada no dataset.")


if __name__ == "__main__":
    main()