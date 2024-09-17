from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import DBSCAN
import numpy as np

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

from scipy.stats import weibull_min

def clustering_kmedoids(joint_emb, num_clusters=10,verbose = False):

    joint_emb_np = joint_emb.cpu().detach().numpy()
    
    pic = KMedoids(n_clusters=num_clusters, random_state=42)

    pic.fit(joint_emb_np)
    cluster_labels = pic.labels_  
    if verbose:
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"Total clusters (including noise): {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples")
    
    return cluster_labels

def clustering_pic(joint_emb, num_clusters=10, num_iterations=100, verbose=False):

    joint_emb_np = joint_emb.cpu().detach().numpy()
    

    similarity_matrix = np.dot(joint_emb_np, joint_emb_np.T)
    

    v = np.random.rand(similarity_matrix.shape[0])
    v = v / np.linalg.norm(v)
    

    for _ in range(num_iterations):
        v = np.dot(similarity_matrix, v)
        v = v / np.linalg.norm(v)
    

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(v.reshape(-1, 1))
    
    if verbose:

        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        

        print(f"Total clusters: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples")
    
    return cluster_labels


def compute_best_mapping(old_labels, new_labels):
    old_length = len(old_labels)
    
    truncated_new_labels = new_labels[:old_length]

    old_classes = np.unique(old_labels)
    new_classes = np.unique(truncated_new_labels)
    cost_matrix = np.zeros((len(old_classes), len(new_classes)))

    for i, old_class in enumerate(old_classes):
        for j, new_class in enumerate(new_classes):
            cost_matrix[i, j] = -np.sum((old_labels == old_class) & (truncated_new_labels == new_class))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {new_classes[j]: old_classes[i] for i, j in zip(row_ind, col_ind)}


    for new_class in new_classes:
        if new_class not in mapping:
            mapping[new_class] = -1  

    return mapping




def clustering(joint_emb,num_clusters = 10):

    joint_emb_np = joint_emb.cpu().detach().numpy()  


    kmeans = KMeans(n_clusters=num_clusters, random_state=0)


    kmeans.fit(joint_emb_np)
    cluster_labels = kmeans.labels_  
    return cluster_labels


def clustering_dbscan(joint_emb, eps=0.5, min_samples=5, verbose = False):

    joint_emb_np = joint_emb.cpu().detach().numpy()


    dbscan = DBSCAN(eps=eps, min_samples=min_samples)


    dbscan.fit(joint_emb_np)
    cluster_labels = dbscan.labels_  
    if verbose:

        unique_labels, counts = np.unique(cluster_labels, return_counts=True)


        print(f"Total clusters (including noise): {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples")
    return cluster_labels


def clustering_with_constraints(joint_emb, eps=0.5, min_samples=5, min_clusters=5, max_clusters=50):

    joint_emb_np = joint_emb.cpu().detach().numpy()
    

    scaler = StandardScaler()
    joint_emb_np = scaler.fit_transform(joint_emb_np)


    dbscan = DBSCAN(eps=eps, min_samples=min_samples)


    cluster_labels = dbscan.fit_predict(joint_emb_np)


    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    num_clusters = len(unique_clusters)

    if num_clusters < min_clusters:
        print(f"Number of clusters ({num_clusters}) is less than the minimum required ({min_clusters}).")

    elif num_clusters > max_clusters:
        print(f"Number of clusters ({num_clusters}) is more than the maximum allowed ({max_clusters}).")

        cluster_centers = np.array([joint_emb_np[cluster_labels == k].mean(axis=0) for k in unique_clusters])
        distances = cdist(cluster_centers, cluster_centers)
        np.fill_diagonal(distances, np.inf)
        while len(unique_clusters) > max_clusters:

            i, j = np.unravel_index(np.argmin(distances), distances.shape)

            cluster_labels[cluster_labels == unique_clusters[j]] = unique_clusters[i]
            unique_clusters = np.unique(cluster_labels[cluster_labels != -1])

            cluster_centers = np.array([joint_emb_np[cluster_labels == k].mean(axis=0) for k in unique_clusters])
            distances = cdist(cluster_centers, cluster_centers)
            np.fill_diagonal(distances, np.inf)

    return cluster_labels



class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ImprovedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes+1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def init_classifier(joint_emb,device,num_clusters=10):
    embedding_dim = joint_emb.shape[1]
    hidden_dim = 128 
    classifier = ImprovedClassifier(input_dim=embedding_dim,hidden_dim=hidden_dim, num_classes=num_clusters)
    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    return classifier,classifier_optimizer,criterion


def train_classifier(joint_emb, cluster_labels, classifier, optimizer, device, lr=0.001, num_epochs=10):

    embedding_dim = joint_emb.shape[1]

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    

    joint_emb_tensor = joint_emb 
    cluster_labels_tensor = torch.tensor(cluster_labels, dtype=torch.long).to(device)


    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        outputs = classifier(joint_emb_tensor)
        loss = criterion(outputs, cluster_labels_tensor)
        loss.backward(retain_graph=True)
        optimizer.step()


        _, predicted = torch.max(outputs, 1)
        correct = (predicted == cluster_labels_tensor).sum().item()
        accuracy = correct / cluster_labels_tensor.size(0)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy*100:.2f}%")
    return classifier, criterion



class OpenMaxClassifier(nn.Module):
    def __init__(self, alpha=10):
        super(OpenMaxClassifier, self).__init__()
        self.alpha = alpha  

    def fit_weibull(self, distances):

        params = weibull_min.fit(distances, floc=0)
        return params

    def get_distances(self, features, mavs):

        distances = torch.cdist(features, mavs, p=2)
        return distances

    def compute_openmax(self, features, mavs, weibull_params):

        distances = self.get_distances(features, mavs)
        weibull_cdf = torch.tensor([weibull_min.cdf(d.detach().cpu().numpy(), *params) for d, params in zip(distances.T, weibull_params)]).T.to(features.device)
        scores = torch.exp(-distances)
        
        # Adjust scores using Weibull CDF
        openmax_scores = scores * weibull_cdf

        # Sum of adjusted scores for known classes
        known_scores_sum = openmax_scores.sum(dim=1, keepdim=True)

        # Calculate unknown class score
        unknown_scores = 1 - known_scores_sum

        # Concatenate known and unknown scores
        openmax_scores = torch.cat([openmax_scores, unknown_scores], dim=1)

        return openmax_scores

    def forward(self, x, mavs, weibull_params):
        openmax_scores = self.compute_openmax(x, mavs, weibull_params)
        return openmax_scores

def init_openmax_classifier(train_features, train_labels, num_classes, device, alpha=10):
    classifier = OpenMaxClassifier(alpha)
    classifier = classifier.to(device)

    mavs = torch.stack([train_features[train_labels == i].mean(dim=0) for i in range(num_classes)], dim=0)
    mavs = mavs.to(device)

    weibull_params = []
    for i in range(num_classes):
        distances = torch.cdist(train_features[train_labels == i], mavs[i:i+1], p=2).view(-1)
        params = classifier.fit_weibull(distances.detach().cpu().numpy())
        weibull_params.append(params)

    return classifier, mavs, weibull_params