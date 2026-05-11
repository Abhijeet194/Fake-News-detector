import os
import re
import string
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


CONFIG = {
    # Data paths
    "fake_path":          "Fake.csv",
    "true_path":          "True.csv",

        # Training parameters
    
    "sample_size":        20000,     

    "test_size":          0.2,
    "random_state":       42,
    "log_path":           "training.log",
    "output_dir":         "bert_fake_news_model"
}


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("FakeNewsDetector")
    logger.setLevel(logging.DEBUG)

    # Console output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))

    # File output — saves everything to training.log
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = setup_logging(CONFIG["log_path"])



#  STEP 1 — LOAD DATA

def load_data(config: dict) -> pd.DataFrame:
    logger.info("=" * 55)
    logger.info("STEP 1: Loading dataset")
    logger.info("=" * 55)

    df = pd.DataFrame()
    try:
        fake_df = pd.read_csv(config["fake_path"])
        true_df = pd.read_csv(config["true_path"])
        fake_df["label"] = 0   # 0 = FAKE
        true_df["label"] = 1   # 1 = REAL


        
        if config["sample_size"] is not None:
            n = config["sample_size"]
            fake_df = fake_df.sample(n, random_state=config["random_state"])
            true_df = true_df.sample(n, random_state=config["random_state"])
       

        fake_df["content"] = fake_df["title"] + " " + fake_df["text"]
        true_df["content"] = true_df["title"] + " " + true_df["text"]

        df = pd.concat([fake_df, true_df], ignore_index=True)
        df = df[["content", "label"]].dropna()

        # Check if the loaded data is too small for stratification
        fake_count = (df.label==0).sum()
        real_count = (df.label==1).sum()

        if fake_count < 2 or real_count < 2:
            logger.warning(f"Insufficient data loaded from CSVs for stratification (Fake: {fake_count}, Real: {real_count}). Generating synthetic data instead.")
            df = generate_synthetic_data(n=8000) # Ensure enough data, e.g., 8000 articles (4000 fake, 4000 real)

    except FileNotFoundError:
        logger.warning("Dataset CSVs not found or are too small — generating synthetic demo data")
        df = generate_synthetic_data(n=8000) # Ensure enough data if files are not found

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=config["random_state"]).reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} articles | Fake: {(df.label==0).sum():,} | Real: {(df.label==1).sum():,}")

    # Check class imbalance
    if (df.label==0).sum() == 0 or (df.label==1).sum() == 0:
        logger.warning("One or more classes have zero samples. Cannot calculate ratio.")
    else:
        ratio = (df.label==0).sum() / (df.label==1).sum()
        if ratio < 0.8 or ratio > 1.2:
            logger.warning(f"Class imbalance detected! Ratio: {ratio:.2f}")
        else:
            logger.info(f"Class balance OK. Fake/Real ratio: {ratio:.2f}")

    return df.reset_index(drop=True)


def generate_synthetic_data(n: int = 2000) -> pd.DataFrame:
    import random
    fake_phrases = [
        "SHOCKING: Government hides truth about",
        "You won't believe what they found",
        "BREAKING: Secret agenda exposed",
        "Mainstream media won't tell you",
        "Anonymous source reveals hidden",
        "Miracle cure suppressed by big pharma",
    ]
    real_phrases = [
        "The government announced new policy on",
        "Scientists published research showing",
        "According to official reports",
        "The committee voted to approve",
        "The central bank released figures on",
        "The study published in the journal found",
    ]
    topics = ["economy", "health", "elections", "climate", "technology", "education"]
    rows = []
    for _ in range(n // 2):
        rows.append({
            "content": f"{random.choice(fake_phrases)} {random.choice(topics)}.",
            "label": 0
        })
        rows.append({
            "content": f"{random.choice(real_phrases)} {random.choice(topics)}.",
            "label": 1
        })
    logger.info(f"Generated {n:,} synthetic articles")
    return pd.DataFrame(rows)



#  STEP 2 — TEXT PREPROCESSING


def preprocess(text: str) -> str:
    # BERT does its own tokenization so we just clean basic things
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    text = re.sub(r"^[A-Z][A-Z\s,]+\s*\([A-Za-z]+\)\s*[-–]\s*", "", text)
    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 55)
    logger.info("STEP 2: Preprocessing text (this takes a few minutes...)")
    logger.info("=" * 55)
    df = df.copy()
    df["clean"] = df["content"].apply(preprocess)
    avg_len = df["clean"].str.split().str.len().mean()
    logger.info(f"Preprocessed {len(df):,} articles | Avg tokens: {avg_len:.0f}")
    return df



#  PREDICT NEW ARTICLES


def predict_bert(text, model, tokenizer):

    
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation     = True,
        padding        = True,
        max_length     = 256
    )

    # Move all input tensors to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred       = torch.argmax(probs).item()
    label      = "REAL " if pred == 1 else "FAKE "
    confidence = probs[0][pred].item()

    print(f"{label} ({confidence*100:.2f}%)")



#  MAIN

def main():
    start = datetime.now()
    logger.info("Fake News Detection Pipeline v2.0 — production Ready")
    logger.info(f"Started: {start.isoformat()}")

    # 1. Load
    df = load_data(CONFIG)

    # 2. Preprocess
    df = preprocess_dataset(df)

    # 3. Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["clean"].tolist(),
        df["label"].tolist(),
        test_size    = 0.2,
        stratify     = df["label"],
        random_state = 42
    )
    logger.info(f"Train: {len(train_texts):,} | Test: {len(test_texts):,}")

    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 5. Tokenization
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=256)

    class NewsDataset(torch.utils.data.Dataset):

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels    = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    # 6. Datasets
    train_dataset = NewsDataset(train_encodings, train_labels)
    test_dataset  = NewsDataset(test_encodings,  test_labels)

    # 7. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2
    )

    # 8. Training setup
    training_args = TrainingArguments(
        output_dir                  = "./results",
        learning_rate               = 1e-5,
        per_device_train_batch_size = 8,        
        per_device_eval_batch_size  = 8,
        num_train_epochs            = 3,
        eval_strategy               = "epoch",
        save_total_limit            = 2,
        save_strategy               = "epoch",
        metric_for_best_model       = "accuracy",
        fp16                        = True,      
        warmup_ratio                = 0.1,       
        weight_decay                = 0.01,      
        seed                        = 42,
    )

    # 9. Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            
            "f1":       f1_score(labels, preds, average="weighted")
        }

    # 10. Trainer
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        compute_metrics = compute_metrics
    )

    # 10.5 Train & Evaluate
    trainer.train()
    trainer.evaluate()

    trainer.save_model("bert_fake_news_model")
    tokenizer.save_pretrained("bert_fake_news_model")

    model.eval()

    # 11. Test predictions
    logger.info("=" * 55)
    logger.info("TEST PREDICTIONS")
    logger.info("=" * 55)

    tests = [
        "Scientists discover vaccine with 95% efficacy published in the New England Journal of Medicine.",
        "SHOCKING: Government secretly puts mind-control chemicals in water supply! Anonymous sources reveal!!!",
        "Donald Trump will distribute 5 gold coins to every beggar on May 6 2026.",
        "The Federal Reserve raised interest rates by 0.25 percentage points citing inflation concerns.",
    ]

    for t in tests:
        predict_bert(t, model, tokenizer)

    elapsed = (datetime.now() - start).seconds
    logger.info(f"Done in {elapsed}s | Model saved to {CONFIG['output_dir']} | Log saved to {CONFIG['log_path']}")


if __name__ == "__main__":
    main()