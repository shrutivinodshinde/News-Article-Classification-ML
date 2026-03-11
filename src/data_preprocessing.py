import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from src import config
from src.visualization import plot_class_distribution

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def preprocess_data():

    print("Loading dataset...")

    df = pd.read_csv(config.RAW_DATA_PATH)

    df = df.dropna()

    print("Cleaning text...")

    df["clean_text"] = df["Text"].apply(clean_text)

    processed = df[["clean_text", "Category"]]

    processed.to_csv(config.PROCESSED_DATA_PATH, index=False)

    print("Processed data saved.")

    plot_class_distribution()