"""IMDB Review Sentiment Classifier.

This console application loads the IMDB review dataset, explores its statistics,
cleans and tokenizes the text, trains a TF-IDF + Multinomial Naive Bayes classifier,
and reports model performance for both a sample subset and the full dataset.
"""

from __future__ import annotations

import io
import re
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import download, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
OUTPUT_PATH = RESULTS_DIR / "output.txt"
RANDOM_SEED = 42
# Negation can reverse sentiment, so these words must survive stop-word filtering.
NEGATION_WORDS = {"not", "no", "never", "nor", "neither", "cannot"}

# Make tokenizer and corpus data available before cleaning any review.
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except LookupError:
    download("punkt", quiet=True)

try:
    import nltk
    nltk.data.find("corpora/stopwords")
except LookupError:
    download("stopwords", quiet=True)

try:
    import nltk
    nltk.data.find("corpora/wordnet")
except LookupError:
    download("wordnet", quiet=True)

try:
    import nltk
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    download("omw-1.4", quiet=True)

# Reuse these NLP resources across reviews instead of rebuilding them per call.
PORTER_STEMMER = PorterStemmer()
WORDNET_LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))


def initialize_nltk_resources() -> None:
    """Ensure required NLTK resources are installed before text processing."""
    resources = [
        ("tokenizers", "punkt"),
        ("tokenizers", "punkt_tab"),
        ("corpora", "stopwords"),
        ("corpora", "wordnet"),
        ("corpora", "omw-1.4"),
    ]
    for resource_type, resource_name in resources:
        try:
            import nltk
            nltk.data.find(f"{resource_type}/{resource_name}")
        except LookupError:
            download(resource_name, quiet=True)


def find_dataset_path(root: Path) -> Path:
    """Locate the IMDB CSV dataset regardless of case or folder naming."""
    candidates = [
        root / "dataset",
        root / "Dataset",
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            for csv_file in candidate.rglob("*.csv"):
                if "imdb" in csv_file.name.lower():
                    return csv_file
    for csv_file in root.rglob("*.csv"):
        if "imdb" in csv_file.name.lower():
            return csv_file
    raise FileNotFoundError("No IMDB dataset CSV file was found under the project root.")


def load_dataset() -> pd.DataFrame:
    """Load the dataset and validate the required columns."""
    dataset_path = find_dataset_path(PROJECT_ROOT)
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded from: {dataset_path}")
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must include 'review' and 'sentiment' columns.")
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Display dimensions, schema, missing values, and class distribution."""
    print("\n=== PHASE 1: DATASET LOADING AND EXPLORATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Total reviews: {len(df)}")
    print("\nColumn names and dtypes:")
    print(df.dtypes.to_string())

    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values.to_string())

    review_missing = df["review"].isnull().sum()
    sentiment_missing = df["sentiment"].isnull().sum()
    if review_missing or sentiment_missing:
        print("\nFilling missing review/sentiment values with empty strings/""Unknown"" for transparent handling.")
        df["review"] = df["review"].fillna("")
        df["sentiment"] = df["sentiment"].fillna("Unknown")

    print("\nUnique sentiment labels:")
    print(sorted(df["sentiment"].dropna().unique().tolist()))

    sentiment_counts = df["sentiment"].value_counts(dropna=False)
    print("\nSentiment counts:")
    print(sentiment_counts.to_string())

    sentiment_percentages = df["sentiment"].value_counts(normalize=True, dropna=False) * 100
    print("\nSentiment percentages:")
    print(sentiment_percentages.round(2).to_string())

    majority_class = sentiment_counts.max() / len(df)
    print(f"\nMajority-class baseline accuracy: {majority_class:.4f}")


def analyze_word_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Create a word-count feature and compute summary statistics."""
    print("\n=== WORD COUNT ANALYSIS ===")
    df = df.copy()
    df["review"] = df["review"].fillna("")
    df["word_count"] = df["review"].str.split().str.len()
    word_summary = df["word_count"].describe()
    print(word_summary.to_string())

    mean_wc = df["word_count"].mean()
    median_wc = df["word_count"].median()
    mode_wc = df["word_count"].mode().iloc[0]
    variance_wc = df["word_count"].var(ddof=1)
    std_wc = df["word_count"].std(ddof=1)
    min_wc = df["word_count"].min()
    max_wc = df["word_count"].max()
    iqr_wc = df["word_count"].quantile(0.75) - df["word_count"].quantile(0.25)
    mode_band_width = max(1, int(np.ceil(2 * iqr_wc / len(df) ** (1 / 3))))
    band_starts = (df["word_count"] // mode_band_width) * mode_band_width
    band_counts = band_starts.value_counts()
    most_frequent_bands = band_counts[band_counts == band_counts.max()].index
    mode_bands = ", ".join(
        f"[{int(start)}, {int(start) + mode_band_width - 1}]"
        for start in sorted(most_frequent_bands)
    )

    print(f"Mean word count: {mean_wc:.2f}")
    print(f"Median word count: {median_wc:.2f}")
    print(f"Mode word count: {mode_wc}")
    print(
        "Mode-band width (Freedman-Diaconis rule, rounded up to whole words): "
        f"{mode_band_width} words"
    )
    print(
        "Most frequent word-count interval(s), inclusive: "
        f"{mode_bands} ({int(band_counts.max())} reviews)"
    )
    print(f"Variance: {variance_wc:.2f}")
    print(f"Standard deviation: {std_wc:.2f}")
    print(f"Minimum word count: {min_wc}")
    print(f"Maximum word count: {max_wc}")

    if mean_wc > median_wc:
        print("Mean exceeds median, suggesting a right-skewed distribution with some very long reviews.")
    elif mean_wc < median_wc:
        print("Median exceeds mean, suggesting a left-skewed distribution or a few short reviews.")
    else:
        print("Mean and median are close, indicating a relatively balanced distribution.")
    return df


def analyze_duplicates(df: pd.DataFrame) -> None:
    """Report duplicate rows without silently removing them."""
    print("\n=== DUPLICATE ANALYSIS ===")
    duplicate_count = int(df.duplicated().sum())
    unique_reviews = df["review"].nunique()
    print(f"Duplicate rows: {duplicate_count}")
    print(f"Unique reviews: {unique_reviews}")


def analyze_sentiment_lengths(df: pd.DataFrame) -> None:
    """Compute sentiment-wise word-count statistics using groupby."""
    print("\n=== SENTIMENT-WISE WORD-LENGTH ANALYSIS ===")
    df = df.copy()
    df["word_count"] = df["review"].fillna("").str.split().str.len()
    grouped = df.groupby("sentiment")["word_count"]
    pos_mean = grouped.get_group("positive").mean()
    neg_mean = grouped.get_group("negative").mean()
    pos_median = grouped.get_group("positive").median()
    neg_median = grouped.get_group("negative").median()

    print(f"Average positive review length: {pos_mean:.2f} words")
    print(f"Average negative review length: {neg_mean:.2f} words")
    print(f"Median positive review length: {pos_median:.2f} words")
    print(f"Median negative review length: {neg_median:.2f} words")
    print("Positive and negative reviews are compared directly; any difference may reflect review style or sentiment intensity.")


def generate_visualizations(df: pd.DataFrame) -> None:
    """Create simple statistical plots and save them under results/plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sentiment_counts = df["sentiment"].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        hue=sentiment_counts.index,
        dodge=False,
        palette="deep",
        legend=False,
    )
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sentiment_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df["word_count"], bins=40, kde=True, color="mediumseagreen")
    plt.title("Review Word-Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "review_length_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="sentiment", y="word_count", hue=df["sentiment"], dodge=False, palette="Set2", legend=False)
    plt.title("Sentiment-wise Review Length Comparison")
    plt.xlabel("Sentiment")
    plt.ylabel("Word Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "review_length_comparison.png", dpi=200)
    plt.close()

    print("\nSaved plots to results/plots/")


def clean_review(text: str) -> str:
    """Clean a single review using HTML stripping, lowercase, tokenization, stop-word filtering, and lemmatization."""
    if pd.isna(text):
        return ""

    cleaned = str(text)
    cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.lower()
    tokens = word_tokenize(cleaned)
    filtered_tokens = []
    for token in tokens:
        if token.isalpha() and (token not in STOP_WORDS or token in NEGATION_WORDS):
            filtered_tokens.append(token)
    stemmed_tokens = [PORTER_STEMMER.stem(token) for token in filtered_tokens]
    lemmatized_tokens = [WORDNET_LEMMATIZER.lemmatize(token, pos="v") for token in filtered_tokens]
    return " ".join(lemmatized_tokens)


def demonstrate_cleaning_example() -> None:
    """Show preprocessing stages on a sample review and explain stemming vs lemmatization."""
    example_text = "<br />I was not happy with the movie, but it was never boring!<br>"
    print("\n=== NLP PREPROCESSING DEMONSTRATION ===")
    print(f"Original text:\n{example_text}")

    html_stripped = re.sub(r"<br\s*/?>", " ", example_text, flags=re.IGNORECASE)
    html_stripped = re.sub(r"<[^>]+>", " ", html_stripped)
    print(f"\nHTML-stripped text:\n{html_stripped}")

    lower_text = html_stripped.lower()
    print(f"\nLowercase text:\n{lower_text}")

    tokens = word_tokenize(lower_text)
    print(f"\nTokenized text:\n{tokens}")

    filtered_tokens = [token for token in tokens if token.isalpha() and (token not in STOP_WORDS or token in NEGATION_WORDS)]
    print(f"\nStop-word-filtered text:\n{filtered_tokens}")

    stemmed_tokens = [PORTER_STEMMER.stem(token) for token in filtered_tokens]
    print(f"\nStemmed text:\n{stemmed_tokens}")

    lemmatized_tokens = [WORDNET_LEMMATIZER.lemmatize(token, pos="v") for token in filtered_tokens]
    print(f"\nLemmatized text:\n{lemmatized_tokens}")

    print("\nDifference between stemming and lemmatization:")
    print("- Stemming is a rule-based reduction that often cuts words to a rough root, such as 'better' -> 'better' or 'movies' -> 'movi'.")
    print("- Lemmatization uses vocabulary and grammar to reduce words to a dictionary form, such as 'better' -> 'good' and 'movies' -> 'movie'.")
    print(
        "Lemmatization is preferred because it converts words into meaningful dictionary forms instead of simply cutting off word endings. "
        "This helps preserve the semantic meaning of reviews and provides more consistent features for sentiment classification."
    )


def benchmark_cleaning(df: pd.DataFrame, sample_size: int = 1000) -> None:
    """Benchmark module-level initialization against repeated local initialization."""
    print("\n=== CLEANING PERFORMANCE BENCHMARK ===")
    reviews = df["review"].head(sample_size).fillna("").tolist()

    start = time.perf_counter()
    _ = [clean_review(review) for review in reviews]
    efficient_time = time.perf_counter() - start

    def reference_cleaning(text: str) -> str:
        local_stemmer = PorterStemmer()
        local_lemmatizer = WordNetLemmatizer()
        local_stop_words = set(stopwords.words("english"))
        text = str(text)
        text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.lower()
        tokens = word_tokenize(text)
        filtered = [token for token in tokens if token.isalpha() and (token not in local_stop_words or token in NEGATION_WORDS)]
        stemmed = [local_stemmer.stem(token) for token in filtered]
        lemmatized = [local_lemmatizer.lemmatize(token, pos="v") for token in filtered]
        return " ".join(lemmatized)

    start = time.perf_counter()
    _ = [reference_cleaning(review) for review in reviews]
    repeated_time = time.perf_counter() - start

    print(f"Efficient initialization outside the function: {efficient_time:.6f} seconds")
    print(f"Repeated initialization inside benchmark reference: {repeated_time:.6f} seconds")


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to every review and save the result in a dedicated column."""
    print("\n=== DATASET CLEANING ===")
    processed = df.copy()
    processed["review"] = processed["review"].fillna("")
    processed["cleaned_review"] = processed["review"].apply(clean_review)
    empty_count = int((processed["cleaned_review"].str.len() == 0).sum())
    successfully_processed = int((processed["cleaned_review"].str.len() > 0).sum())
    print(f"Reviews that become empty after cleaning: {empty_count}")
    print(f"Successfully processed reviews: {successfully_processed}")
    return processed


def prepare_sample_dataset(df: pd.DataFrame, sample_size: int = 20000) -> pd.DataFrame:
    """Build a balanced 20k subset with 10k positive and 10k negative reviews."""
    # Sample each class equally so the initial experiment is balanced and reproducible.
    pos_reviews = df[df["sentiment"] == "positive"].sample(n=sample_size // 2, random_state=RANDOM_SEED, replace=False)
    neg_reviews = df[df["sentiment"] == "negative"].sample(n=sample_size // 2, random_state=RANDOM_SEED, replace=False)
    sample_df = pd.concat([pos_reviews, neg_reviews], ignore_index=True)
    print(f"\nPrepared balanced sample dataset with {len(sample_df)} rows.")
    return sample_df


def train_and_evaluate(X_train: pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series, label_names: list[str], run_name: str) -> dict:
    """Train a TF-IDF + Multinomial Naive Bayes model and evaluate it."""
    print(f"\n=== {run_name} ===")
    # Both experiments share this function so their vectorizer and classifier settings stay identical.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=50000,
        sublinear_tf=True,
    )
    # Fit vocabulary and IDF on training text only; test reviews must not leak into feature construction.
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training feature shape: {X_train_vec.shape}")
    print(f"Testing feature shape: {X_test_vec.shape}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="binary", pos_label="positive")
    recall = recall_score(y_test, predictions, average="binary", pos_label="positive")
    f1 = f1_score(y_test, predictions, average="binary", pos_label="positive")
    report = classification_report(y_test, predictions, target_names=label_names)
    cm = confusion_matrix(y_test, predictions, labels=label_names)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("\nConfusion matrix:")
    print(cm)

    cm_labels = ["Negative", "Positive"] if label_names == ["negative", "positive"] else label_names
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title(f"{run_name} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
        "model": model,
        "vectorizer": vectorizer,
        "label_names": label_names,
    }


def extract_important_terms(model: MultinomialNB, vectorizer: TfidfVectorizer, label_names: list[str]) -> Tuple[list[str], list[str]]:
    """Return the strongest positive and negative terms from the learned log-probabilities."""
    vocab = vectorizer.get_feature_names_out()
    log_probs = model.feature_log_prob_
    pos_index = label_names.index("positive") if "positive" in label_names else 1
    neg_index = label_names.index("negative") if "negative" in label_names else 0

    pos_term_scores = log_probs[pos_index] - log_probs[neg_index]
    neg_term_scores = log_probs[neg_index] - log_probs[pos_index]

    top_positive = [vocab[i] for i in np.argsort(pos_term_scores)[::-1][:20]]
    top_negative = [vocab[i] for i in np.argsort(neg_term_scores)[::-1][:20]]

    return top_positive, top_negative


def save_results(output_text: str) -> None:
    """Persist the project output to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        output_file.write(output_text)


def main() -> None:
    """Run all dataset-loading, analysis, preprocessing, model training, and reporting tasks."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    initialize_nltk_resources()
    raw_df = load_dataset()
    raw_df["sentiment"] = raw_df["sentiment"].astype(str).str.strip().str.lower()
    raw_df["review"] = raw_df["review"].fillna("")
    explore_dataset(raw_df)
    raw_df = analyze_word_counts(raw_df)
    analyze_duplicates(raw_df)
    analyze_sentiment_lengths(raw_df)
    generate_visualizations(raw_df)
    demonstrate_cleaning_example()
    benchmark_cleaning(raw_df)

    cleaned_df = preprocess_dataset(raw_df)
    sample_df = prepare_sample_dataset(cleaned_df)

    X_sample = sample_df["cleaned_review"].fillna("")
    y_sample = sample_df["sentiment"].astype(str).str.lower()
    # Stratification keeps the balanced sample's class proportions in both splits.
    X_sample_train, X_sample_test, y_sample_train, y_sample_test = train_test_split(
        X_sample,
        y_sample,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_sample,
    )
    sample_results = train_and_evaluate(
        X_sample_train,
        X_sample_test,
        y_sample_train,
        y_sample_test,
        ["negative", "positive"],
        "INITIAL 20,000-REVIEW EXPERIMENT",
    )
    valid_df = cleaned_df[(cleaned_df["cleaned_review"].notna()) & (cleaned_df["cleaned_review"].str.len() > 0)].copy()
    print(f"\nUsing {len(valid_df)} valid records for full-dataset experiment.")

    X_full = valid_df["cleaned_review"].fillna("")
    y_full = valid_df["sentiment"].astype(str).str.lower()
    # Stratification preserves the full dataset's class proportions in both splits.
    X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_full,
    )
    full_results = train_and_evaluate(
        X_full_train,
        X_full_test,
        y_full_train,
        y_full_test,
        ["negative", "positive"],
        "FULL DATASET EXPERIMENT",
    )
    # Report terms learned from the final full-dataset model, not the sample model.
    top_positive, top_negative = extract_important_terms(
        full_results["model"],
        full_results["vectorizer"],
        full_results["label_names"],
    )
    print("\n20 strongest positive terms from the FULL-DATASET MultinomialNB model:")
    print(top_positive)
    print("\n20 strongest negative terms from the FULL-DATASET MultinomialNB model:")
    print(top_negative)

    print(f"\nSample-dataset accuracy: {sample_results['accuracy']:.4f}")
    print(f"Full-dataset accuracy: {full_results['accuracy']:.4f}")
    print("\nThis comparison shows how the same TF-IDF + Multinomial Naive Bayes pipeline performs on a smaller balanced subset and on the full dataset.")


if __name__ == "__main__":
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main()
    output = buffer.getvalue()
    complete_output = output + f"\nSaved terminal output to {OUTPUT_PATH}\n"
    save_results(complete_output)
    print(complete_output, end="")
