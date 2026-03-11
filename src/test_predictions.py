import joblib
import re
import nltk
from nltk.corpus import stopwords
from src import config

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def run_test_predictions():

    model, vectorizer, _, _ = joblib.load(config.MODEL_PATH)

    test_sentences = [

        "Apple released a new artificial intelligence chip for smartphones",

        "The prime minister addressed parliament regarding the upcoming election",

        "Manchester United won the football match after scoring two late goals",

        "The company reported record profits in the stock market this quarter",

        "The latest Hollywood movie broke box office records worldwide",

        "Researchers developed a new software platform for cloud computing",

        "The government introduced new tax policies for businesses",

        "The tennis champion won the grand slam final after a thrilling match",

        "A famous actor announced a new film project",

        "Technology companies are investing heavily in artificial intelligence"

    ]

    output_path = config.BASE_DIR + "/results/test_predictions.txt"

    with open(output_path, "w") as f:

        f.write("NEWS ARTICLE CLASSIFICATION TEST RESULTS\n")
        f.write("=======================================\n\n")

        for sentence in test_sentences:

            cleaned = clean_text(sentence)

            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)[0]

            result = f"Sentence: {sentence}\nPredicted Category: {prediction}\n\n"

            print(result)

            f.write(result)

    print("Test predictions saved to results/test_predictions.txt")


if __name__ == "__main__":

    run_test_predictions()