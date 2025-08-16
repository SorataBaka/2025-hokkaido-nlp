import numpy as np
from torch import i0
from load_dataset import process_dataset
from create_buckets import create_buckets
from build_dataset import build_dataset_bucketed, calculate_label_ratio
from tokenizer_functions import get_tokens
from train_model import train
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# first import and the dataset from huggingface

print("Loading and processing dataset...")
training_dataset_reduced, test_dataset_reduced = process_dataset(split_size=0.01)
print(f"Dataset loaded and processed. Loaded {len(training_dataset_reduced)} training samples and {len(test_dataset_reduced)} test samples.")

print("Loading embeddings...")
embeddings = np.load("bert_mean_pooled_embeddings.npy")
print("Loaded embeddings with shape:", embeddings.shape)

# Start bucketing
print("Starting bucketing process...")
training_dataset_reduced, unique_labels, labels = create_buckets(training_dataset_reduced, embeddings, min_cluster_size=200)

positive_ratio, negative_ratio = calculate_label_ratio(training_dataset_reduced, "label")

grouped_datasets = [
  training_dataset_reduced.select(indices) for label in unique_labels for indices in [np.where(labels == label)[0]]
]

print(f"Created {len(grouped_datasets)} buckets based on HDBSCAN clustering.")

print("Building final dataset...")

for i in range(2100, training_dataset_reduced.num_rows, 100):
  print(f"Processing {i} samples...")
  final_train_dataset, final_test_dataset = build_dataset_bucketed(grouped_datasets, i, positive_ratio, negative_ratio)

  print(f"Final dataset built. Training set size: {len(final_train_dataset)}, Test set size: {len(final_test_dataset)}")

  final_train_dataset = final_train_dataset.map(get_tokens, batched=True, batch_size=8)
  final_test_dataset = final_test_dataset.map(get_tokens, batched=True, batch_size=8)
  test_dataset_reduced = test_dataset_reduced.map(get_tokens, batched=True, batch_size=8)

  if "labels" not in final_train_dataset.column_names:
    final_train_dataset = final_train_dataset.rename_column("label", "labels")
  if "labels" not in final_test_dataset.column_names:
    final_test_dataset = final_test_dataset.rename_column("label", "labels")
  if "labels" not in test_dataset_reduced.column_names:
    test_dataset_reduced = test_dataset_reduced.rename_column("label", "labels")
  print("Tokenization complete. Starting training...")

  train(final_train_dataset, final_test_dataset, test_dataset_reduced, "Bucketing")