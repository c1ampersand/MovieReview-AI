import os
import tarfile
import urllib.request
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = "data"
TAR_PATH = os.path.join(DATA_DIR, "aclImdb_v1.tar.gz")
EXTRACTED_DIR = os.path.join(DATA_DIR, "aclImdb")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACTED_DIR):
        print("IMDB dataset already downloaded.")
        return

    print("Downloading IMDB dataset. This may take a few minutes...")

    urllib.request.urlretrieve(DATA_URL, TAR_PATH)

    print("Extracting dataset...")

    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    print("Dataset ready.")


def load_reviews(split, max_per_class):
    texts = []
    labels = []

    for label_name, label_value in [("neg", 0), ("pos", 1)]:
        folder = os.path.join(EXTRACTED_DIR, split, label_name)
        files = os.listdir(folder)[:max_per_class]

        for filename in files:
            file_path = os.path.join(folder, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            texts.append(text)
            labels.append(label_value)

    return texts, labels


print("Preparing dataset...")
download_dataset()

print("Loading reviews...")

X_train, y_train = load_reviews("train", max_per_class=5000)
X_test, y_test = load_reviews("test", max_per_class=1500)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("Training model...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("Testing model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"],
    output_dict=True
)

conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
print("Confusion Matrix:")
print(conf_matrix)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/movie_review_model.pkl")

metrics = {
    "accuracy": accuracy,
    "negative_precision": report["Negative"]["precision"],
    "negative_recall": report["Negative"]["recall"],
    "negative_f1": report["Negative"]["f1-score"],
    "positive_precision": report["Positive"]["precision"],
    "positive_recall": report["Positive"]["recall"],
    "positive_f1": report["Positive"]["f1-score"],
    "true_negative": int(conf_matrix[0][0]),
    "false_positive": int(conf_matrix[0][1]),
    "false_negative": int(conf_matrix[1][0]),
    "true_positive": int(conf_matrix[1][1])
}

pd.DataFrame([metrics]).to_csv("data/model_metrics.csv", index=False)

sample_reviews = pd.DataFrame({
    "review": [
        "The movie was amazing and emotional with excellent acting.",
        "The film was boring, slow, and badly written.",
        "I loved the characters and the story was beautiful.",
        "The plot made no sense and the acting was terrible."
    ],
    "expected_sentiment": [
        "Positive",
        "Negative",
        "Positive",
        "Negative"
    ]
})

sample_reviews.to_csv("data/sample_reviews.csv", index=False)

print("Model saved to models/movie_review_model.pkl")
print("Metrics saved to data/model_metrics.csv")
print("Sample reviews saved to data/sample_reviews.csv")
print("Training complete.")