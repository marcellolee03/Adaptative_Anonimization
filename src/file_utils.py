import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

FILE_CSV_2 = "cic_iot2.csv"
FILE_CSV_8 = "cic_iot8.csv" 
FILE_CSV_34 = "cic_iot34.csv"

def consolidate_labels(dataset):
    """
    Consolida todas as colunas 'label_*' em uma única coluna 'Label'.
    O valor será o nome do ataque (sem o prefixo 'label_').
    """
    # Identificar todas as colunas que começam com 'label_'
    label_columns = [col for col in dataset.columns if col.startswith('label_')]
    
    # Criar uma nova coluna 'Label' baseada nas colunas 'label_*'
    dataset['Label'] = 'Unknown'
    
    # Para cada linha, encontrar qual coluna 'label_*' tem valor 1 (ou o maior valor)
    for i, row in dataset.iterrows():
        for label_col in label_columns:
            if row[label_col] == 1:  # Assumindo que as labels são one-hot encoded
                # Remover o prefixo 'label_' e definir como valor da coluna 'Label'
                attack_name = label_col.replace('label_', '')
                dataset.at[i, 'Label'] = attack_name
                break
    
    # Remover as colunas originais 'label_*'
    dataset = dataset.drop(columns=label_columns)
    
    return dataset

def remove_source_file_columns(dataset):
    """
    Remove colunas que começam com 'source_file_' que não são relevantes para a análise.
    """
    source_columns = [col for col in dataset.columns if col.startswith('source_file_')]
    if source_columns:
        dataset = dataset.drop(columns=source_columns)
    return dataset

def handle_missing_values(dataset):
    """
    Trata valores ausentes (NaN) no dataset.
    
    - Para colunas numéricas: substitui com a mediana
    - Para colunas categóricas: substitui com o valor mais frequente
    
    Returns:
        DataFrame: Dataset com valores NaN tratados
    """
    # Verificar se existem valores ausentes
    nan_count = dataset.isna().sum().sum()
    if nan_count > 0:
        print(f"Encontrados {nan_count} valores ausentes no dataset.")
        
        # Identificar colunas numéricas e categóricas
        numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = dataset.select_dtypes(include=['object', 'bool', 'category']).columns
        
        # Imputação para colunas numéricas (usando mediana)
        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            dataset[numeric_cols] = num_imputer.fit_transform(dataset[numeric_cols])
        
        # Imputação para colunas categóricas (usando valor mais frequente)
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                # Se a coluna tiver valores ausentes
                if dataset[col].isna().any():
                    # Obter o valor mais frequente (excluindo NaN)
                    most_frequent = dataset[col].value_counts().index[0]
                    # Substituir NaN pelo valor mais frequente
                    dataset[col].fillna(most_frequent, inplace=True)
        
        # Verificar se ainda existem valores ausentes
        remaining_nan = dataset.isna().sum().sum()
        if remaining_nan > 0:
            print(f"Atenção: Ainda restam {remaining_nan} valores ausentes.")
            # Para garantir que não haja valores NaN, podemos remover as linhas restantes
            dataset.dropna(inplace=True)
            print(f"Removidas {nan_count - remaining_nan} linhas com valores ausentes.")
        else:
            print("Todos os valores ausentes foram tratados com sucesso.")
    else:
        print("Não foram encontrados valores ausentes no dataset.")
    
    return dataset

def get_file():
    """
    Get dataset file and apply preprocessing:
    - Consolidate label columns
    - Handle missing values (NaN)
    - Apply MinMaxScaler to numeric features
    - Apply LabelEncoder to non-numeric features (mantendo mesmas colunas)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the absolute path of the file
    file_path = os.path.join(script_dir, '..', 'data', FILE_CSV_34)
    print(f"Carregando arquivo: {file_path}")
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the CSV
        dataset = pd.read_csv(file_path)
        print(f"Dataset original: {dataset.shape}")
        
        # Remover colunas de arquivo fonte
        dataset = remove_source_file_columns(dataset)
        
        # Consolidar colunas de label
        dataset = consolidate_labels(dataset)
        print(f"Após consolidação de labels: {dataset.shape}")
        
        # Tratar valores ausentes
        dataset = handle_missing_values(dataset)
        
        # Identificar targets
        if 'Label' in dataset.columns:
            # Converter Label para números para processos de ML
            # Preservar o mapeamento para referência futura
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(dataset['Label'])
            
            # Criar mapeamento de labels para valores numéricos
            label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            print(f"Labels encontradas: {len(label_mapping)}")
            print(f"Mapeamento de labels: {label_mapping}")
            
            # Salvar mapeamento para uso posterior
            mapping_df = pd.DataFrame(list(label_mapping.items()), columns=['Label', 'Index'])
            mapping_path = os.path.join(script_dir, '..', 'results', 'label_mapping.csv')
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            mapping_df.to_csv(mapping_path, index=False)
            print(f"Mapeamento de labels salvo em: {mapping_path}")
            
            # Remover a coluna Label do dataset
            X_df = dataset.drop('Label', axis=1)
        else:
            y = None
            X_df = dataset
            print("Aviso: Coluna 'Label' não encontrada.")
        
        # Identificar numeric and non-numeric columns (excluindo Label)
        numeric_cols = X_df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_df.select_dtypes(include=['object', 'bool', 'category']).columns
        
        print(f"Colunas numéricas: {len(numeric_cols)}")
        print(f"Colunas categóricas: {len(categorical_cols)}")
        
        # Aplicar RobustScaler nas features numéricas
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])
        
        # Aplicar LabelEncoder nas features categóricas
        encoders = {}
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                encoders[col] = LabelEncoder()
                X_df[col] = encoders[col].fit_transform(X_df[col])
                print(f"Coluna {col}: {len(encoders[col].classes_)} valores únicos")
                
            # Salvar mapeamento dos encoders para referência
            encoder_mappings = {}
            for col, encoder in encoders.items():
                encoder_mappings[col] = {label: idx for idx, label in enumerate(encoder.classes_)}
            
            # Salvar em um arquivo para referência
            encoder_path = os.path.join(script_dir, '..', 'results', 'encoder_mappings.csv')
            pd.DataFrame([encoder_mappings]).to_csv(encoder_path, index=False)
            print(f"Mapeamento dos encoders salvo em: {encoder_path}")
        
        print(f"Dataset final: {X_df.shape}")
        
        # Verificar se há valores infinitos e substituí-los
        if X_df.isin([np.inf, -np.inf]).any().any():
            print("Atenção: Foram encontrados valores infinitos. Substituindo por valores grandes finitos.")
            X_df.replace([np.inf, -np.inf], [1e10, -1e10], inplace=True)
        
        # Reconstruir o dataset completo
        if y is not None:
            # Adicionar o y transformado de volta como coluna 'Label'
            final_dataset = X_df.copy()
            final_dataset['Label'] = y
            return final_dataset
        else:
            return X_df
    
    print(f"Erro: Arquivo {file_path} não encontrado.")
    return None