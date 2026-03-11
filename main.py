from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():

    print("NEWS ARTICLE CLASSIFICATION PIPELINE\n")

    preprocess_data()

    train_model()

    evaluate_model()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()