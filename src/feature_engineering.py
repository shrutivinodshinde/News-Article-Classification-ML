import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src import config


def create_features():

    print("Generating TF-IDF features...")

    df = pd.read_csv(config.PROCESSED_DATA_PATH)

    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["clean_text"])

    y = df["Category"]

    return X, y, vectorizer