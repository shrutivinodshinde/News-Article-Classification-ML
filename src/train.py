import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from src.feature_engineering import create_features
from src.visualization import plot_model_comparison
from src import config


def train_model():

    X, y, vectorizer = create_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "NaiveBayes": MultinomialNB(),
        "LinearSVM": LinearSVC()
    }

    results = {}

    best_model = None
    best_score = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        results[name] = acc

        print(name, "accuracy:", acc)

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print("Best model:", best_name)

    plot_model_comparison(results)

    # Hyperparameter tuning
    param_grid = {"C":[0.1,1,5,10]}

    grid = GridSearchCV(best_model, param_grid, cv=5)

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    joblib.dump((best_model, vectorizer, X_test, y_test),
                config.MODEL_PATH)