print("Importing Dependencies")
import ast
import os
import numpy as np
import pandas as pd
import torch
import ast

from safetensors.torch import save_file
from datasets import Dataset, DatasetDict, Sequence, Value, Features
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.metrics import mean_squared_error, accuracy_score



# ──────── CONSTANTS ──────── #
EMO_ORDER = ["joy", "sadness", "anticipation", "surprise",
             "anger", "fear", "disgust", "trust"]
BASE_MODEL = "model/bert-base-japanese-v3"
DEFAULT_OUTPUT_DIR = "model/sentiment-bert/"
DATAPATH = 'data/wrime/'
CSV_SPLITS = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
TWEET_PATH = "data/tweets/clustered/zero_shot/"
OUT_PATH = "data/tweets/sentiments/"

# ──────── HELPERS ──────── #
def dict_to_vec(d):
    if isinstance(d, str):
        d = ast.literal_eval(d)
    return [int(d[e]) for e in EMO_ORDER]

def load_wrime_dataset(target_col="writer") -> DatasetDict:
    def load_split(path, target_col="writer"):
        df = pd.read_csv(path)
        df["labels"] = df[target_col].apply(dict_to_vec)
        return Dataset.from_pandas(df[["sentence", "labels"]])

    dataset = {k: load_split(DATAPATH+v) for k, v in CSV_SPLITS.items()}
    return DatasetDict(dataset)

def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    def hf_tokenize(batch, tokenizer):
        out = tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        out["labels"] = [list(map(float, lab)) for lab in batch["labels"]]
        return out

    # ...
    tokenized = dataset.map(
        hf_tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["sentence"],
    )

    for split in tokenized.keys():
        features = Features({
            **tokenized[split].features,
            "labels": Sequence(Value("float32"))
        })
        tokenized[split] = tokenized[split].cast(features)

    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.rint(logits).clip(0, 3).astype(int).flatten()
    labels = labels.flatten()
    return {
        "mse": mean_squared_error(labels, logits.flatten()),
        "accuracy": accuracy_score(labels, preds)
    }

def train_wrime_model(
    target_col="writer",
    output_dir=DEFAULT_OUTPUT_DIR,
    epochs=30,
    batch_size=16,
    lr=5e-5,
    seed=42
):
    print("▶ Loading data...")
    dataset = load_wrime_dataset(target_col=target_col)

    print("▶ Loading tokenizer and base model...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    data_collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=8,
            problem_type="regression"
    )
    model.config.problem_type = "regression"



    print("▶ Tokenizing dataset...")
    tokenized = tokenize_dataset(dataset, tokenizer)

    print("▶ Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type = "cosine",
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=seed,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Model device:", trainer.model.device)
    print("▶ Starting training...")
    trainer.train()
    print("✅ Training complete.")

    return trainer, tokenizer

def save_model(trainer, tokenizer, output_dir=DEFAULT_OUTPUT_DIR):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    weights = {k: v.cpu() for k, v in trainer.model.state_dict().items()}

    # Save safetensors weights file
    save_file(weights, f"{output_dir}/pytorch_model.safetensors")

def predict(text, model_path=DEFAULT_OUTPUT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, ignore_mismatched_sizes=True)
    model.eval()

    batch = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**batch).logits.squeeze(0).tolist()
    scores = [round(max(0, min(3, s)), 2) for s in logits]
    return {k: round(v, 2) for k, v in dict(zip(EMO_ORDER, scores)).items()}

def emotion_space_transformation(emotion_dict):
    # Russell-space coordinates for Plutchik emotions
    mapping = {
        "joy":          (0.6, 0.6),
        "trust":        (0.6, 0.2),
        "fear":        (-0.7, 0.6),
        "surprise":     (0.0, 0.8),
        "sadness":     (-0.8, 0.1),
        "disgust":     (-0.6, 0.4),
        "anger":       (-0.8, 0.7),
        "anticipation": (0.3, 0.6),
    }

    total_weight = 0
    valence_sum = 0
    arousal_sum = 0

    for emotion, raw_score in emotion_dict.items():
        if emotion in mapping:
            weight = raw_score / 4.0  # normalize to [0, 1]
            valence, arousal = mapping[emotion]
            valence_sum += valence * weight
            arousal_sum += arousal * weight
            total_weight += weight

    if total_weight == 0:
        return (0.0, 0.0)

    return (arousal_sum / total_weight, valence_sum / total_weight)

if __name__ == "__main__":
    training = False
    testing = False
    overwrite = False

    annotators  = [ 
        'writer', 
        'reader1', 
        'reader2', 
        'reader3', 
        'avg_readers'
    ]

    usernames = [
        "ecoyuri",
        "shinji_ishimaru",
        "renho_sha",
        "toshio_tamogami",
    ]

    if training:
        for annotator in annotators:
            trainer, tokenizer = train_wrime_model(target_col=annotator)
            save_model(trainer, tokenizer, f"{DEFAULT_OUTPUT_DIR}sentiment-bert-{annotator}")
    elif testing:
        while True:
            print("Please enter a Japanese sentence: ", end="")
            text = input()
            if text in ['break', 'exit', 'stop', '']: break
            for annotator in annotators:
                result = predict(text, f"{DEFAULT_OUTPUT_DIR}sentiment-bert-{annotator}")
                print(f"{result} - {annotator}")
    else:
        for username in usernames:
            df = None
            if overwrite:
                print(f"Generating Sentiment Prediction with sentiment-bert-{annotator}")
                print(f"▶ Loading Tweets from {TWEET_PATH}{username}_tweets.csv")
                df = pd.read_csv(f"{TWEET_PATH}{username}_tweets.csv")
                for annotator in annotators:
                    df[annotator] = df['tweet'].apply(
                        lambda tweet: predict(tweet, f"{DEFAULT_OUTPUT_DIR}sentiment-bert-{annotator}")
                    )
            else:
                print(f"▶ Loading Tweets from {TWEET_PATH}{username}_tweets.csv")
                df = pd.read_csv(f"{OUT_PATH}{username}_tweets.csv")
                for annotator in annotators: df[annotator] = df[annotator].apply(ast.literal_eval)
                    
                for annotator in annotators:
                    averages = []
                    russell = []
                    total = None
                    for idx, row in df.iterrows():
                        total = {'joy': 0, 'sadness': 0, 'anticipation': 0, 'surprise': 0, 'anger': 0, 'fear': 0, 'disgust': 0, 'trust': 0}
                        for annotator in annotators:
                            for k, v in row[annotator].items():
                                total[k] += float(v)
                        average = {k: round(float(v) / len(annotators), 2) for k, v in total.items()}
                        averages.append(str(average))
                        arousal, valance = emotion_space_transformation(average)
                        russell.append(str({'arousal': arousal, 'valance': valance}))
                    df['averages'] = averages
                    df['russell'] = russell
            
            print(f"▶ Saving results to {OUT_PATH}")
            df.to_csv(f"{OUT_PATH}{username}_tweets.csv", index=False)
