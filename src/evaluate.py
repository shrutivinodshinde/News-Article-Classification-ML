import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.visualization import plot_confusion_matrix, plot_top_words
from src import config


def evaluate_model():

    model, vectorizer, X_test, y_test = joblib.load(config.MODEL_PATH)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    cm = confusion_matrix(y_test, preds)

    report = classification_report(y_test, preds)

    labels = list(set(y_test))

    plot_confusion_matrix(cm, labels)

    plot_top_words(vectorizer, model)

    with open(config.METRICS_PATH, "w") as f:

        f.write("Accuracy: " + str(acc) + "\n\n")

        f.write("Confusion Matrix\n")

        f.write(str(cm))

        f.write("\n\n")

        f.write(report)

    print("Final Accuracy:", acc)