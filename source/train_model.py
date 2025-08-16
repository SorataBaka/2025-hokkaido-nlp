from tensorflow.python.training import evaluation
import torch
import pandas as pd
import shutil
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from transformers import (
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
    default_data_collator,
)

excel_path = "results.xlsx"
sheet_name = "results"
CLASS_NAMES = ["negative", "positive"]

# Detect MPS
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Training arguments
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    eval_strategy="epoch",
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=3e-5,
    lr_scheduler_type="linear",
    dataloader_pin_memory=False,
    bf16=True,  # Enable mixed precision training
    report_to="none",  # Disable reporting to external services
)

# Metrics computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # ensure logits and labels are on CPU for NumPy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    preds = logits.argmax(axis=1)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
    }
    for idx, name in enumerate(CLASS_NAMES):
        metrics[f"{name}_precision"] = p_cls[idx]
        metrics[f"{name}_recall"]    = r_cls[idx]
        metrics[f"{name}_f1"]        = f1_cls[idx]
        metrics[f"{name}_support"]   = support_cls[idx]
    return metrics

# Evaluation function
def evaluate(model, eval_dataset, train_amount, method_name):
    evaluation_trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    print("Starting evaluation...")
    pred_out = evaluation_trainer.predict(eval_dataset)
    manual_metrics = compute_metrics((pred_out.predictions, pred_out.label_ids))

    row = {
        "method": method_name,
        "stage": 1,
        "data_amount": train_amount,
        "eval_loss": pred_out.metrics.get("test_loss", None),
        "eval_accuracy": manual_metrics["accuracy"],
        "eval_macro_precision": manual_metrics["macro_precision"],
        "eval_macro_recall": manual_metrics["macro_recall"],
        "eval_macro_f1": manual_metrics["macro_f1"],
        "eval_negative_precision": manual_metrics["negative_precision"],
        "eval_negative_recall": manual_metrics["negative_recall"],
        "eval_negative_f1": manual_metrics["negative_f1"],
        "eval_positive_precision": manual_metrics["positive_precision"],
        "eval_positive_recall": manual_metrics["positive_recall"],
        "eval_positive_f1": manual_metrics["positive_f1"],
    }

    df = pd.DataFrame([row])

    # Load existing Excel or create new
    try:
        existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df = pd.concat([existing_df, df], ignore_index=True)
    except (FileNotFoundError, ValueError):
        pass

    df.to_excel(excel_path, index=False, sheet_name=sheet_name)

# Training function
def train(train_dataset, test_dataset, eval_dataset, method_name):
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased",
        cache_dir="caches/",
        num_labels=len(CLASS_NAMES),
    )
    model.to(device)
    print(f"Training model on {device}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        optimizers=(AdamW(model.parameters(), lr=5e-5), None),
    )
    trainer.train()

    # Evaluate after training
    evaluate(model, eval_dataset, len(train_dataset) + len(test_dataset), method_name)
    
    # Delete saved progress
    if os.path.exists("./trainer_output"):
        shutil.rmtree("./trainer_output")
    print("Training complete. Model saved and evaluation metrics recorded.")
    
