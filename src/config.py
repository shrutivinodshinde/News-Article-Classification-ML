import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw/BBC News Train.csv")

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/processed_data.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models/news_classifier.pkl")

METRICS_PATH = os.path.join(BASE_DIR, "results/metrics.txt")

TEST_SIZE = 0.2
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 10000