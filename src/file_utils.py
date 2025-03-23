import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

FILE_CSV_ADULTS = "adults.csv"
FILE_CSV_DDOS = "ddos_csv_100000.csv" 
FILE_CSV_HEART = "heart.csv"
FILE_CSV_CMC = "cmc.csv"
FILE_CSV_MGM = "mgm.csv"
FILE_CSV_CAHOUSING = "cahousing.csv"

def get_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the absolute path of the file
    file_path = os.path.join(script_dir, '..', 'data', FILE_CSV_CAHOUSING)
    print(f"Caminho do arquivo: {file_path}")
    
    # Check if the files exists
    if os.path.exists(file_path):
        # Read the CSV
        dataset = pd.read_csv(file_path)
        
        # Imprimir número de colunas
        num_colunas = len(dataset.columns)
        print(f"Número de colunas no dataset: {num_colunas}")
        
        # Imprimir nomes das colunas
        print("Colunas do dataset:")
        for i, coluna in enumerate(dataset.columns):
            print(f"  {i+1}. {coluna}")
        
        # Verificação de colunas não numéricas para aplicar Label Encoder
        print("\nAplicando Label Encoder nas colunas não numéricas:")
        for column in dataset.columns:
            if column != 'Label' and dataset[column].dtype == 'object':
                print(f"  Aplicando Label Encoder na coluna: {column}")
                le = LabelEncoder()
                dataset[column] = le.fit_transform(dataset[column])
        
        return dataset