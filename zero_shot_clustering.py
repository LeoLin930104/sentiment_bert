from __future__ import annotations
from typing import List
print("Importing Dependencies")
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models, util
from tqdm import tqdm

# ==========================================================================
# Configuration — edit these constants as needed
# ==========================================================================
TWEETS_DIR: str  = "data/tweets/preprocessed/" # folder containing one or more tweet CSVs
TOPICS_CSV: str  = "data/VAA/vaaQuestions.csv" # topic definitions
OUT_CSV: str     = "data/tweets/clustered/zero_shot/"
MODEL_NAME: str  = "sonoisa/sentence-bert-base-ja-mean-tokens-v2" # any Sentence‑Transformers model
BATCH_SIZE: int  = 128
THRESHOLD: float = 0   # min cosine similarity; below → topic = "unknown"
MAX_SEQ_LEN: int = 256    # truncate long tweets if using HF base models

# ==========================================================================
# Helper functions
# ==========================================================================

def build_st_model(model_name: str, device: str) -> SentenceTransformer:
    """Return a SentenceTransformer model, wrapping with mean pooling if needed."""
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        print(f"⚙️  Wrapping '{model_name}' with mean pooling …")
        word_emb = models.Transformer(model_name, max_seq_length=MAX_SEQ_LEN)
        pooling = models.Pooling(
            word_emb.get_word_embedding_dimension(), pooling_mode="mean"
        )
        return SentenceTransformer(modules=[word_emb, pooling], device=device)
    
def embed(sentences: List[str], model: SentenceTransformer, batch_size: int):
    return model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    usernames = [
        "ecoyuri",
        "shinji_ishimaru",
        "renho_sha",
        "toshio_tamogami",
    ]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = build_st_model(MODEL_NAME, device)

    print("Loading & constructing topic sentences …")
    topics = pd.read_csv(TOPICS_CSV)
    topic_sentences = [f"{row.Topic}、{row.Question}　{row.Details}" for row in topics.itertuples(index=False)]
    topics = [row.Topic for row in topics.itertuples(index=False)]
    print(f"Loaded {len(topic_sentences)} topic sentences")

    print("Embedding topic sentences …")
    topic_emb = embed(topic_sentences, model, BATCH_SIZE)
    
    for username in usernames:

        print("Loading tweets …")
        tweets_df = pd.read_csv(f"{TWEETS_DIR}{username}_tweets.csv")
        print(f"Loaded {len(tweets_df):,} tweets from {TWEETS_DIR}")

        print("Embedding tweets …")
        tweet_emb = embed(tweets_df["tweet"].tolist(), model, BATCH_SIZE)

        print("Computing cosine similarities …")
        cosine_scores = util.cos_sim(tweet_emb, topic_emb)  # (n_tweets, n_topics)
        best_scores, best_idx = cosine_scores.max(dim=1)

        print("Assigning topics …")
        tweets_df["topic_idx"] = best_idx.cpu().numpy()
        tweets_df["topic"] = [topics[i] for i in best_idx]
        tweets_df["similarity"] = best_scores.cpu().numpy()

        if THRESHOLD is not None:
            mask = tweets_df["similarity"] < THRESHOLD
            tweets_df.loc[mask, ["topic_idx", "topic"]] = (-1, "unknown")

        tweets_df = tweets_df.sort_values(by='similarity', ascending=False)

        tweets_df.to_csv(f"{OUT_CSV}{username}_tweets.csv", index=False, encoding="utf-8")
        print(f"✔ Results written to {OUT_CSV}")
