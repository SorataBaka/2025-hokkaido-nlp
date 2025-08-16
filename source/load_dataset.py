from datasets import load_dataset
import re
def clean_text(row):
  text = row["text"]
  # Remove HTML tags
  text = re.sub(r"<.*?>", " ", text)
  # Remove non-printable characters
  text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
  # Replace multiple spaces/newlines with single space
  text = re.sub(r"\s+", " ", text)
  # Optionally lowercase
  text = text.strip()  # Don't lowercase if case matters
  return {
      "text": text
  }

def process_dataset(split_size=0.01):
  original_dataset = load_dataset("mteb/amazon_polarity", cache_dir="caches/")
  original_dataset = original_dataset.with_format("torch")
  training_dataset = original_dataset["train"] # type: ignore
  test_dataset = original_dataset["test"] # type: ignore

  training_dataset_split = training_dataset.train_test_split(test_size=split_size, seed=42) # type: ignore
  test_dataset_split = test_dataset.train_test_split(test_size=split_size, seed=42) # type: ignore

  training_dataset_reduced = training_dataset_split["test"]
  test_dataset_reduced = test_dataset_split["test"]

  training_dataset_reduced = training_dataset_reduced.map(clean_text)
  test_dataset_reduced = test_dataset_reduced.map(clean_text)
  
  return training_dataset_reduced, test_dataset_reduced