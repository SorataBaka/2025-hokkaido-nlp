import numpy as np
from torch import i0
from load_dataset import process_dataset
from build_dataset import build_dataset_random
from tokenizer_functions import get_tokens
from train_model import train
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# first import and the dataset from huggingface

print("Loading and processing dataset...")
training_dataset_reduced, test_dataset_reduced = process_dataset(split_size=0.01)
print(f"Dataset loaded and processed. Loaded {len(training_dataset_reduced)} training samples and {len(test_dataset_reduced)} test samples.")

print("Building final dataset...")

for i in range(2800, 6100, 100):
  print(f"Processing {i} samples...")
  final_train_dataset, final_test_dataset = build_dataset_random(training_dataset_reduced, i)

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

  train(final_train_dataset, final_test_dataset, test_dataset_reduced, "Randomized")