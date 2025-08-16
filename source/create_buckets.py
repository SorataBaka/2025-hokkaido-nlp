from umap import UMAP
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
import numpy as np

def create_buckets(dataset, embeddings, min_cluster_size=200):
  scaler = StandardScaler()
  umap_50 = UMAP(n_components=50, random_state=42)
  umap_2 = UMAP(n_components=2, random_state=42)
  clusterer = HDBSCAN(min_cluster_size=min_cluster_size)

  reduced_embedding = umap_50.fit_transform(embeddings)
  reduced_embedding = umap_2.fit_transform(reduced_embedding)
  reduced_embedding = scaler.fit_transform(reduced_embedding) # type: ignore

  labels = clusterer.fit_predict(reduced_embedding)
  labels = np.array(labels)
  unique_labels = np.unique(labels)
  # Apply the labels on the training dataset

  dataset = dataset.add_column("bucket", labels)
  
  return dataset, unique_labels, labels