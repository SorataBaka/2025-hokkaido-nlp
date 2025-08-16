from datasets import concatenate_datasets
from tokenizer_functions import get_tokens
import math

def calculate_label_ratio(dataset, field):
  #Calculate global ratios
  positive_count = dataset.filter(lambda x: x[field] == 1).num_rows
  negative_count = dataset.filter(lambda x: x[field] == 0).num_rows

  positive_ratio = positive_count / (positive_count + negative_count)
  negative_ratio = negative_count / (positive_count + negative_count)
  
  return positive_ratio, negative_ratio
  
def build_dataset_random(dataset, total_size):
    # Randomly sample from the dataset
    sampled_dataset = dataset.shuffle(seed=42).select(range(total_size))
    sampled_dataset = sampled_dataset.map(get_tokens, batched=True, batch_size=8)
    sampled_dataset = sampled_dataset.rename_column("label", "labels")
    sampled_dataset = sampled_dataset.train_test_split(test_size=0.15, seed=42)
    return sampled_dataset["train"], sampled_dataset["test"]

def build_dataset_bucketed(dataset_groups, total_size, positive_ratio, negative_ratio):
    total_original_size = sum(len(ds) for ds in dataset_groups)
    merged = []
    
    for ds in dataset_groups:
        bucket_share = len(ds) / total_original_size
        bucket_target_size = math.floor(bucket_share * total_size)
        
        positive_filtered = ds.filter(lambda x: x["label"] == 1)
        negative_filtered = ds.filter(lambda x: x["label"] == 0)
        
        max_positive = len(positive_filtered)
        max_negative = len(negative_filtered)
        
        amount_positive = min(math.floor(bucket_target_size * positive_ratio), max_positive)
        amount_negative = min(math.floor(bucket_target_size * negative_ratio), max_negative)
        
        positive_sample = positive_filtered.shuffle(seed=42).select(range(amount_positive))
        negative_sample = negative_filtered.shuffle(seed=42).select(range(amount_negative))
        
        merged.extend([positive_sample, negative_sample])
        
    final_dataset = concatenate_datasets(merged)
    final_dataset = final_dataset.map(get_tokens, batched=True, batch_size=8)
    final_dataset = final_dataset.rename_column("label", "labels")
    final_dataset = final_dataset.train_test_split(test_size=0.15, seed=42)
    return final_dataset["train"], final_dataset["test"]
