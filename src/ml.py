import time

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from anonimization import anonimization_clustering


def cross_validate_k_fold(X, y, anon_training, anon_test, model, model_name, n_clusters, epsilon=1.0):
    kf = StratifiedKFold(n_splits=3)
    scaler = MinMaxScaler()  # Substituído StandardScaler por MinMaxScaler

    accuracy, precision, recall, f1 = [], [], [], []
    training_times, inference_times = [], []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Verificar e corrigir NaN antes da anonimização
        if np.isnan(X_train).any():
            print(f"NaN encontrado em X_train antes da anonimização. Substituindo por 0.")
            X_train = np.nan_to_num(X_train, nan=0.0)
        
        if np.isnan(X_test).any():
            print(f"NaN encontrado em X_test antes da anonimização. Substituindo por 0.")
            X_test = np.nan_to_num(X_test, nan=0.0)

        if anon_training:
            try:
                X_train, y_train = anonimization_clustering(X_train, y_train, n_clusters, epsilon)
                # Verificar NaN após anonimização
                if np.isnan(X_train).any():
                    print(f"NaN encontrado após anonimizar X_train. Substituindo por 0.")
                    X_train = np.nan_to_num(X_train, nan=0.0)
            except Exception as e:
                print(f"Erro durante anonimização do conjunto de treino: {str(e)}")
                print("Continuando sem anonimização para este fold.")

        if anon_test:
            try:
                X_test, y_test = anonimization_clustering(X_test, y_test, n_clusters, epsilon)
                # Verificar NaN após anonimização
                if np.isnan(X_test).any():
                    print(f"NaN encontrado após anonimizar X_test. Substituindo por 0.")
                    X_test = np.nan_to_num(X_test, nan=0.0)
            except Exception as e:
                print(f"Erro durante anonimização do conjunto de teste: {str(e)}")
                print("Continuando sem anonimização para este fold.")

        # Normalizar os dados usando MinMaxScaler
        try:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Verificar NaN após scaling
            if np.isnan(X_train).any():
                print(f"NaN encontrado após scaling de X_train. Substituindo por 0.")
                X_train = np.nan_to_num(X_train, nan=0.0)
            
            if np.isnan(X_test).any():
                print(f"NaN encontrado após scaling de X_test. Substituindo por 0.")
                X_test = np.nan_to_num(X_test, nan=0.0)
                
            # Verificar valores infinitos também
            if np.isinf(X_train).any():
                print(f"Valores infinitos encontrados em X_train. Substituindo por valores grandes finitos.")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                
            if np.isinf(X_test).any():
                print(f"Valores infinitos encontrados em X_test. Substituindo por valores grandes finitos.")
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
        except Exception as e:
            print(f"Erro durante normalização: {str(e)}")
            print("Tentando continuar sem normalização...")
            # Último recurso para garantir dados limpos
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Medir tempo de treinamento
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Medir tempo de inferência/detecção
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred, average='weighted'))
            recall.append(recall_score(y_test, y_pred, average='weighted'))
            f1.append(f1_score(y_test, y_pred, average='weighted'))
        except Exception as e:
            print(f"Erro durante treinamento/teste do modelo {model_name}: {str(e)}")
            # Adicionar valores de fallback para não interromper o experimento
            training_times.append(0)
            inference_times.append(0)
            accuracy.append(0)
            precision.append(0)
            recall.append(0)
            f1.append(0)

    # Se não conseguimos coletar métricas em nenhum fold, retornar valores padrão
    if not accuracy:
        print(f"ALERTA: Não foi possível coletar métricas para o modelo {model_name}.")
        return [anon_training, anon_test, 0, 0, 0, 0, 0, 0]

    results = {
        'accuracy': np.array(accuracy), 
        'precision': np.array(precision),
        'recall': np.array(recall), 
        'f1_score': np.array(f1),
        'training_time': np.array(training_times),
        'inference_time': np.array(inference_times)
    }

    print(model_name, anon_training, anon_test)
    for k in results.keys():
        print(f"{k} ---> mean: {results[k].mean():.4f}, std: {results[k].std():.4f}")

    return [
        anon_training, 
        anon_test, 
        results['accuracy'].mean(), 
        results['precision'].mean(), 
        results['recall'].mean(), 
        results['f1_score'].mean(),
        results['training_time'].mean(),
        results['inference_time'].mean()
    ]