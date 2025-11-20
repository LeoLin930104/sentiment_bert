print("Importing Dependencies")
from keybert import KeyBERT
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from fugashi import Tagger
import pandas as pd
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

# === Config ===
TWEETS_DIR = "data/tweets/preprocessed/"  # CSV with tweet texts
TWEETS_COLUMN = "tweet"
TOPICS_CSV = "data/VAA/vaaQuestions.csv"
KEYBERT_MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"

# === Helper: Japanese tokenizer ===
def mecab_nouns(text: str):
    tagger = Tagger()
    return [w.surface for w in tagger(text) if w.pos.startswith("名詞")]

# === Step 1: Extract seed keywords from topic descriptions ===
def extract_topic_seeds():
    topics = pd.read_csv(TOPICS_CSV)
    topic_sentences = [f"{row.Topic}、{row.Question}　{row.Details}" for row in topics.itertuples(index=False)]

    kw_model = KeyBERT(KEYBERT_MODEL_NAME)
    vectorizer = CountVectorizer(
        tokenizer=mecab_nouns,
        ngram_range=(1, 2),
        min_df=1,
        lowercase=False,
        stop_words=None
    )

    topic_seeds = []
    for sentence in topic_sentences:
        keywords = kw_model.extract_keywords(
            sentence,
            vectorizer=vectorizer,
            keyphrase_ngram_range=(1, 3),
            top_n=5,
            use_mmr=True,
            diversity=0.3
        )
        topic_seeds.append([kw for kw, _ in keywords])
    return topic_seeds

# === Step 2: Load tweets ===
def load_tweets():
    usernames = [ "ecoyuri", "shinji_ishimaru", "renho_sha", "toshio_tamogami"]
    tweets = []
    for username in usernames:
        df = pd.read_csv(f"{TWEETS_DIR}{username}_tweets.csv")
        tweets = tweets + df[TWEETS_COLUMN].tolist()
    return tweets

# === Step 3: Run BERTopic with seed words ===
def run_bertopic_with_seeds(tweets, topic_seeds):
    # Flatten seed list
    flat_seed_words = [kw for topic in topic_seeds for kw in topic]

    # Boost seed terms in topic modeling
    ctfidf_model = ClassTfidfTransformer(
        seed_words=flat_seed_words,
        seed_multiplier=5.0
    )

    embed_model = SentenceTransformer(KEYBERT_MODEL_NAME)
    topic_model = BERTopic(
        embedding_model=embed_model,
        ctfidf_model=ctfidf_model,
        language="multilingual",
        verbose=True
    )

    topics, probs = topic_model.fit_transform(tweets)
    return topic_model, topics, probs

# === Main execution ===
if __name__ == "__main__":
    print("Extracting seed keywords …")
    topic_seeds = extract_topic_seeds()
    print(f"Extracted {len(topic_seeds[0])} keywords for {len(topic_seeds)} topics.")

    print("Loading tweets …")
    tweets = load_tweets()
    print(f"Extracted {len(tweets)} tweets.")

    print("Running BERTopic with seed words …")
    topic_model, topics, probs = run_bertopic_with_seeds(tweets, topic_seeds)
    print(topic_model.get_topic(1))
    # Print top 10 topics
    print(topic_model.get_topic_info().head(10))

    # Optional: save model or results
    # topic_model.save("model/bertopic_semi_supervised/bertopic_semi_supervised")
