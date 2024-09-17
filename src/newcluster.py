import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import torch
def plot_k_distance(joint_emb, min_samples=5):

    joint_emb_np = joint_emb.cpu().detach().numpy()

    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(joint_emb_np)
    distances, indices = nbrs.kneighbors(joint_emb_np)


    k_distances = np.sort(distances[:, min_samples-1])


    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    plt.title('k-Distance Graph')
    plt.show()
    plt.savefig('haha.png')

def clustering(joint_emb, eps=0.5, min_samples=5):

    joint_emb_np = joint_emb.cpu().detach().numpy()


    dbscan = DBSCAN(eps=eps, min_samples=min_samples)


    dbscan.fit(joint_emb_np)
    cluster_labels = dbscan.labels_  

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    print(f"Total clusters (including noise): {len(unique_labels)}")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples")

    return cluster_labels

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluate_stability(labels1, labels2):
    min_length = min(len(labels1), len(labels2))
    labels1_half = labels1[:min_length]
    labels2_half = labels2[:min_length]
    ari = adjusted_rand_score(labels1_half, labels2_half)
    nmi = normalized_mutual_info_score(labels1_half, labels2_half)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    return ari, nmi


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(labels1, labels2):
    conf_matrix = confusion_matrix(labels1, labels2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Cluster Labels 2')
    plt.ylabel('Cluster Labels 1')
    plt.title('Confusion Matrix of Clustering Results')
    plt.show()
    plt.savefig("confusion_matrix.png")

def plot_jaccard_index_heatmap(labels1, labels2):
    conf_matrix = confusion_matrix(labels1, labels2)
    
    jaccard_index = conf_matrix / (conf_matrix.sum(axis=1)[:, None] + conf_matrix.sum(axis=0)[None, :] - conf_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(jaccard_index, annot=True, cmap='Blues', cbar=True)
    plt.xlabel('Cluster Labels 2')
    plt.ylabel('Cluster Labels 1')
    plt.title('Jaccard Index Heatmap of Clustering Results')
    plt.show()
    plt.savefig("jaccard.png")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_clustering_results_2d(data, labels1, labels2, method='tsne',epoch=0):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2)
    else:
        raise ValueError("Method should be 'pca' or 'tsne'")
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    data_2d = reducer.fit_transform(data)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels1, cmap='viridis', alpha=0.5)
    plt.title('Clustering Result 1')
    
    plt.subplot(1, 2, 2)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels2, cmap='viridis', alpha=0.5)
    plt.title('Clustering Result 2')
    
    plt.show()
    plt.savefig("clustering_compare"+str(epoch)+".png")