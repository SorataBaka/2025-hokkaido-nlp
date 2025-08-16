from transformers import AutoTokenizer # type: ignore
import torch

if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="caches/")

def get_tokens(batch):
  tokens = tokenizer(
    batch["text"],
    truncation=True,
    padding="max_length",
    return_tensors="pt",
    max_length=256
  )
  tokens = {key: value.to(device) for key, value in tokens.items()}
  return tokens