import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import time


def anonimization(data, threshold=0.95, add_noise=True):
    """
    Anonimização de dados usando PCA orientada a eficiência com amostragem diferencial
    
    Args:
        data: Dados a serem anonimizados
        threshold: Limiar de variância explicada para selecionar componentes (padrão: 0.95)
        add_noise: Se deve adicionar ruído diferencial (padrão: True)
    
    Returns:
        Dados anonimizados
    """
    start_time = time.time()
    
    # Tratamento para datasets pequenos ou com poucas features
    n_samples, n_features = data.shape
    min_dim = min(n_samples, n_features)
    
    if min_dim <= 2:
        print(f"Dataset muito pequeno (shape={data.shape}), pulando PCA e aplicando apenas ruído")
        # Se dataset muito pequeno, aplicar apenas ruído e rotação simples
        data_noisy = data.copy()
        if add_noise:
            noise_scale = np.std(data, axis=0) * 0.01
            noise = np.random.normal(0, noise_scale, size=data.shape)
            data_noisy += noise
        return data_noisy
    
    # Normalização dos dados (opcional, mas melhora estabilidade)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Calcular a média de cada coluna
    mean = np.mean(data_scaled, axis=0)
    
    # Centralizar dados
    data_centered = data_scaled - mean
    
    # Aplicar PCA para eficiência - usar número fixo de componentes ao invés de threshold
    # Corrigindo o problema: não pode usar threshold com svd_solver='randomized'
    n_components = max(1, min(min_dim - 1, int(min_dim * 0.8)))  # 80% dos componentes ou no máximo min_dim-1
    print(f"Usando {n_components} componentes PCA de {min_dim} possíveis")
    
    pca = PCA(n_components=n_components, svd_solver='randomized' if min_dim > 10 else 'full')
    data_transformed = pca.fit_transform(data_centered)
    
    # Inovação: Rotação pseudoaleatória dos componentes principais
    # Usa uma matriz de rotação com propriedades de preservação de distância
    n_components = data_transformed.shape[1]
    
    # Gerar matriz de rotação ortogonal aleatória via método QR
    random_matrix = np.random.randn(n_components, n_components)
    q, r = np.linalg.qr(random_matrix)
    
    # Garantir que a matriz tenha determinante positivo (preserva orientação)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    
    # Aplicar rotação
    data_rotated = np.dot(data_transformed, q)
    
    # Adicionar ruído diferencial adaptativo
    if add_noise:
        # Inovação: Ruído diferencial com sensibilidade adaptativa
        # Calcula escala de ruído baseada na distribuição dos dados
        noise_scale = np.std(data_rotated, axis=0) * 0.01
        noise = np.random.normal(0, noise_scale, size=data_rotated.shape)
        data_rotated += noise
    
    # Retornar para espaço original usando componentes rotacionados
    # Usa a matriz de componentes principais do PCA para reconstrução eficiente
    data_original_dimension = np.dot(data_rotated, pca.components_)
    
    # Descentralizar
    data_original_dimension += mean
    
    # Inverter escala
    data_original_dimension = scaler.inverse_transform(data_original_dimension)
    
    end_time = time.time()
    print(f"Tempo de anonimização: {end_time - start_time:.4f} segundos")
    
    return data_original_dimension


def find_clusters(X, k, random_state=42):
    """
    Encontra clusters nos dados
    
    Args:
        X: Dados para clusterização
        k: Número de clusters
        random_state: Seed para reprodutibilidade
    
    Returns:
        Rótulos dos clusters
    """
    # Tratamento para o caso em que k é maior que o número de amostras
    n_samples = X.shape[0]
    k = min(k, n_samples)
    
    # Usar inicialização kmeans++ para convergência mais rápida
    # Ajustando o n_init para evitar warnings em versões recentes do scikit-learn
    Kmean = KMeans(n_clusters=k, n_init='auto', random_state=random_state, init='k-means++')
    Kmean.fit(X)
    return Kmean.labels_


def anonimization_clustering(data, y, k, method='efficient'):
    """
    Anonimização por clusterização
    
    Args:
        data: Dados para anonimização
        y: Rótulos
        k: Número de clusters
        method: Método de anonimização ('original' ou 'efficient')
    
    Returns:
        Dados anonimizados e rótulos correspondentes
    """
    start_time = time.time()
    
    # Verificar se temos dados suficientes
    if data.shape[0] < 3:
        print("Dataset muito pequeno para clustering, retornando dados originais")
        return data, y
    
    # Ajustar k se necessário
    k = min(k, data.shape[0] // 2)
    k = max(k, 1)  # Pelo menos 1 cluster
    
    # Inovação: Clusterização estratificada baseada nas classes
    # Isso garante melhor preservação da relação entre features e targets
    unique_classes = np.unique(y)
    
    # Se temos muitas classes, podemos agrupar por faixas
    if len(unique_classes) > 5:
        print("Muitas classes, agrupando por faixas...")
        # Classificação convencional
        clusters = find_clusters(data, k)
    else:
        # Clusterização estratificada por classe
        print(f"Usando clusterização estratificada para {len(unique_classes)} classes")
        clusters = np.zeros(len(data), dtype=np.int32)
        cluster_offset = 0
        
        for class_val in unique_classes:
            # Índices para esta classe
            class_indices = np.where(y == class_val)[0]
            
            if len(class_indices) < 3:
                # Se não houver amostras suficientes, usar todos como um cluster
                clusters[class_indices] = cluster_offset
                cluster_offset += 1
            else:
                # Clusterizar dentro da classe
                class_data = data[class_indices]
                n_clusters = max(1, min(k, len(class_indices) // 50))  # Ajuste dinâmico
                n_clusters = max(1, min(n_clusters, len(class_indices) // 2))  # Pelo menos 2 samples por cluster
                
                class_clusters = find_clusters(class_data, n_clusters)
                # Atribuir cluster IDs com offset
                clusters[class_indices] = class_clusters + cluster_offset
                cluster_offset += n_clusters
    
    # Organizar índices por cluster
    indices = {}
    for i in range(len(clusters)):
        if clusters[i] not in indices.keys():
            indices[clusters[i]] = []
        indices[clusters[i]].append(i)
    
    data_anonymized, y_in_new_order = None, None
    
    # Inovação: Paralelização por blocos para clusters grandes
    large_clusters = []
    small_clusters = []
    
    # Separar clusters grandes e pequenos
    for k_id in indices.keys():
        if len(indices[k_id]) > 1000:  # Limiar arbitrário, ajuste conforme necessário
            large_clusters.append(k_id)
        else:
            small_clusters.append(k_id)
    
    print(f"Processando {len(small_clusters)} clusters pequenos e {len(large_clusters)} clusters grandes")
    
    # Processar clusters pequenos primeiro
    for k_id in small_clusters:
        cluster_data = data[indices[k_id]]
        if cluster_data.shape[0] < 3:
            # Clusters muito pequenos não passam por anonimização
            anonymized_cluster = cluster_data
        else:
            anonymized_cluster = anonimization(cluster_data)
            
        if data_anonymized is None and y_in_new_order is None:
            data_anonymized = anonymized_cluster
            y_in_new_order = y[indices[k_id]]
        else:
            data_anonymized = np.concatenate((data_anonymized, anonymized_cluster), axis=0)
            y_in_new_order = np.concatenate((y_in_new_order, y[indices[k_id]]), axis=0)
    
    # Processar clusters grandes com otimização específica
    for k_id in large_clusters:
        # Para clusters grandes, aplicar anonimização com configuração mais agressiva
        cluster_data = data[indices[k_id]]
        anonymized_cluster = anonimize_in_batches(cluster_data, batch_size=500)
        
        if data_anonymized is None and y_in_new_order is None:
            data_anonymized = anonymized_cluster
            y_in_new_order = y[indices[k_id]]
        else:
            data_anonymized = np.concatenate((data_anonymized, anonymized_cluster), axis=0)
            y_in_new_order = np.concatenate((y_in_new_order, y[indices[k_id]]), axis=0)
    
    end_time = time.time()
    print(f"Tempo total de anonimização por clustering: {end_time - start_time:.4f} segundos")
    
    return data_anonymized, y_in_new_order


# Função auxiliar para anonimização incremental (processamento por blocos para grandes conjuntos)
def anonimize_in_batches(data, batch_size=10000):
    """
    Anonimiza dados em lotes para conjuntos muito grandes
    
    Args:
        data: Dados para anonimização
        batch_size: Tamanho do lote
    
    Returns:
        Dados anonimizados
    """
    n_samples = data.shape[0]
    
    if n_samples <= batch_size:
        return anonimization(data)
    
    # Processar em lotes
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = data[start_idx:end_idx]
        batches.append(anonimization(batch_data))
    
    # Concatenar resultados
    return np.concatenate(batches, axis=0)