import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src import config


def plot_class_distribution():

    df = pd.read_csv(config.PROCESSED_DATA_PATH)

    plt.figure(figsize=(8,5))

    df["Category"].value_counts().plot(kind="bar")

    plt.title("Class Distribution")

    path = os.path.join(config.BASE_DIR, "results/class_distribution.png")

    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(cm, labels):

    plt.figure(figsize=(8,6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = os.path.join(config.BASE_DIR, "results/confusion_matrix.png")

    plt.savefig(path)
    plt.close()


def plot_model_comparison(results):

    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8,5))

    sns.barplot(x=names, y=scores)

    plt.title("Model Comparison Accuracy")

    path = os.path.join(config.BASE_DIR, "results/model_comparison.png")

    plt.savefig(path)
    plt.close()


def plot_top_words(vectorizer, model):

    feature_names = np.array(vectorizer.get_feature_names_out())

    if not hasattr(model, "coef_"):
        return

    coefs = model.coef_

    top_words = 10

    fig, axes = plt.subplots(coefs.shape[0], 1, figsize=(8,20))

    for i, axis in enumerate(axes):

        top = np.argsort(coefs[i])[-top_words:]

        axis.barh(feature_names[top], coefs[i][top])

    path = os.path.join(config.BASE_DIR, "results/top_words.png")

    plt.tight_layout()

    plt.savefig(path)

    plt.close()