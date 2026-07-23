"""
feature_extraction.py

Generate TF-IDF features from cleaned NLP security prompts (e.g. output of
preprocessing.py) and persist the fitted vectorizer and feature matrix to disk.

Pipeline steps:
    1. Load cleaned dataset (output of preprocessing.py)
    2. Fit a TfidfVectorizer (max_features=10000, ngram_range=(1,2), English stopwords)
    3. Transform the cleaned text into a TF-IDF feature matrix
    4. Save the fitted vectorizer -> vectorizer.pkl
    5. Save the feature matrix    -> X_features.pkl
    6. Return (X, y, vectorizer) for downstream use

Usage:
    python feature_extraction.py

    or as a module:
    from feature_extraction import run_feature_extraction
    X, y, vectorizer = run_feature_extraction(
        "dataset_cleaned.csv", text_column="cleaned_text", label_column="label"
    )
"""

import logging

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_cleaned_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a cleaned dataset CSV into a pandas DataFrame.

    Args:
        filepath: Path to the cleaned CSV file (output of preprocessing.py).

    Returns:
        The loaded DataFrame.
    """
    logger.info(f"Loading cleaned dataset from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def get_labels(df: pd.DataFrame, label_column: str) -> pd.Series:
    """
    Extract the label column from the DataFrame.

    Args:
        df: Input DataFrame.
        label_column: Name of the column containing class labels.

    Returns:
        Series of labels, aligned with df's index.

    Raises:
        KeyError: If label_column is not present in df. This dataset does not
            currently carry labels (they were stripped out when the CSV was
            first built), so this will raise until a label column is added
            back to the input file, or a different label_column is supplied.
    """
    if label_column not in df.columns:
        raise KeyError(
            f"Label column '{label_column}' not found in dataset columns "
            f"{list(df.columns)}. This dataset currently has no label column — "
            f"add one to the CSV (or pass the correct label_column) before "
            f"calling get_labels()."
        )
    return df[label_column]


# --------------------------------------------------------------------------- #
# TF-IDF feature extraction
# --------------------------------------------------------------------------- #

def build_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    stop_words: str = "english",
) -> TfidfVectorizer:
    """
    Construct a TfidfVectorizer with the given configuration.

    Args:
        max_features: Maximum size of the vocabulary (top terms by TF-IDF term frequency).
        ngram_range: Range of n-gram sizes to extract, e.g. (1, 2) for unigrams + bigrams.
        stop_words: Stopword list to use ('english' to remove English stopwords).

    Returns:
        An unfitted TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )


def fit_transform_features(vectorizer: TfidfVectorizer, texts: pd.Series):
    """
    Fit the vectorizer on the given texts and transform them into a TF-IDF matrix.

    Args:
        vectorizer: An (unfitted) TfidfVectorizer instance.
        texts: Series of cleaned text strings.

    Returns:
        A tuple of (X, vectorizer):
            X: sparse TF-IDF feature matrix (scipy.sparse.csr_matrix).
            vectorizer: the now-fitted TfidfVectorizer.
    """
    logger.info(
        f"Fitting TF-IDF vectorizer (max_features={vectorizer.max_features}, "
        f"ngram_range={vectorizer.ngram_range}) on {len(texts)} documents"
    )
    X = vectorizer.fit_transform(texts.astype(str))
    logger.info(f"Generated feature matrix of shape {X.shape}")
    return X, vectorizer


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str) -> None:
    """
    Persist a fitted TfidfVectorizer to disk with joblib.

    Args:
        vectorizer: Fitted TfidfVectorizer instance.
        filepath: Destination path (e.g. 'vectorizer.pkl').
    """
    joblib.dump(vectorizer, filepath)
    logger.info(f"Saved vectorizer to {filepath}")


def save_features(X, filepath: str) -> None:
    """
    Persist a TF-IDF feature matrix to disk with joblib.

    Args:
        X: TF-IDF feature matrix (sparse or dense).
        filepath: Destination path (e.g. 'X_features.pkl').
    """
    joblib.dump(X, filepath)
    logger.info(f"Saved feature matrix to {filepath}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_feature_extraction(
    input_path: str,
    text_column: str = "cleaned_text",
    label_column: str = "label",
    vectorizer_path: str = "vectorizer.pkl",
    features_path: str = "X_features.pkl",
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
):
    """
    Run the complete TF-IDF feature extraction pipeline end-to-end:
    load -> build vectorizer -> fit/transform -> save vectorizer -> save features.

    Args:
        input_path: Path to the cleaned dataset CSV (output of preprocessing.py).
        text_column: Name of the column containing cleaned text.
        label_column: Name of the column containing class labels.
        vectorizer_path: Output path for the fitted vectorizer.
        features_path: Output path for the TF-IDF feature matrix.
        max_features: Maximum vocabulary size for the vectorizer.
        ngram_range: N-gram range for the vectorizer.

    Returns:
        A tuple of (X, y, vectorizer):
            X: TF-IDF feature matrix.
            y: labels Series (None if label_column is not present).
            vectorizer: the fitted TfidfVectorizer.
    """
    df = load_cleaned_dataset(input_path)

    vectorizer = build_vectorizer(max_features=max_features, ngram_range=ngram_range)
    X, vectorizer = fit_transform_features(vectorizer, df[text_column])

    save_vectorizer(vectorizer, vectorizer_path)
    save_features(X, features_path)

    if label_column in df.columns:
        y = get_labels(df, label_column)
    else:
        logger.warning(
            f"Label column '{label_column}' not found in {input_path} — "
            f"returning y=None. Add a label column to the dataset to get labels back."
        )
        y = None

    return X, y, vectorizer


if __name__ == "__main__":
    run_feature_extraction(
        input_path="dataset_cleaned.csv",
        text_column="cleaned_text",
        label_column="label",
        vectorizer_path="vectorizer.pkl",
        features_path="X_features.pkl",
    )
