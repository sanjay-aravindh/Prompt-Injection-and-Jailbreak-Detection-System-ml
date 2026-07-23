"""
preprocessing.py

Text preprocessing pipeline for an NLP security dataset
(e.g. prompt-injection / jailbreak detection datasets with a `text` column).

Pipeline steps:
    1. Load dataset.csv
    2. Remove duplicate prompts
    3. Handle missing values
    4. Convert text to lowercase
    5. Remove URLs
    6. Remove HTML tags
    7. Remove punctuation
    8. Remove numbers
    9. Remove extra spaces
    10. Tokenize text
    11. Remove stopwords
    12. Apply lemmatization
    13. Save cleaned dataset

Usage:
    python preprocessing.py

    or as a module:
    from preprocessing import run_pipeline
    df = run_pipeline("dataset.csv", "dataset_cleaned.csv", text_column="text")
"""

import re
import string
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def download_nltk_resources() -> None:
    """
    Download the NLTK resources required by this pipeline, if not already
    present. Safe to call multiple times (NLTK skips resources it already has).
    """
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        The loaded DataFrame.
    """
    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


# --------------------------------------------------------------------------- #
# Row-level cleaning (duplicates / missing values)
# --------------------------------------------------------------------------- #

def remove_duplicates(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """
    Remove duplicate rows based on a given column (e.g. the text/prompt column).

    Args:
        df: Input DataFrame.
        subset: Column name to check for duplicates.

    Returns:
        DataFrame with duplicate rows removed, index reset.
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} duplicate rows (based on '{subset}')")
    return df


def handle_missing_values(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """
    Drop rows with missing or empty values in the given column.

    Args:
        df: Input DataFrame.
        subset: Column name to check for missing/empty values.

    Returns:
        DataFrame with missing/empty rows removed, index reset.
    """
    before = len(df)
    df = df.dropna(subset=[subset])
    # also drop rows that are empty strings or whitespace-only after stripping
    df = df[df[subset].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} rows with missing/empty '{subset}' values")
    return df


# --------------------------------------------------------------------------- #
# Text-level cleaning functions (each operates on a single string)
# --------------------------------------------------------------------------- #

def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove URLs (http/https/www links) from text."""
    url_pattern = re.compile(r"(https?://\S+|www\.\S+)")
    return url_pattern.sub(" ", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags (e.g. <br>, <div>...</div>) from text."""
    html_pattern = re.compile(r"<.*?>")
    return html_pattern.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove digits from text."""
    return re.sub(r"\d+", " ", text)


def remove_extra_spaces(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Apply the full string-level cleaning pipeline to a single text value:
    lowercase -> remove URLs -> remove HTML tags -> remove punctuation ->
    remove numbers -> remove extra spaces.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    text = str(text)
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_spaces(text)
    return text


# --------------------------------------------------------------------------- #
# Tokenization / stopwords / lemmatization
# --------------------------------------------------------------------------- #

def tokenize_text(text: str) -> list:
    """
    Tokenize a cleaned text string into a list of word tokens.

    Args:
        text: Cleaned text string.

    Returns:
        List of tokens.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list, stop_words: set) -> list:
    """
    Remove English stopwords from a list of tokens.

    Args:
        tokens: List of word tokens.
        stop_words: Set of stopwords to filter out.

    Returns:
        List of tokens with stopwords removed.
    """
    return [tok for tok in tokens if tok not in stop_words]


def lemmatize_tokens(tokens: list, lemmatizer: WordNetLemmatizer) -> list:
    """
    Apply lemmatization to a list of tokens.

    Args:
        tokens: List of word tokens.
        lemmatizer: An initialized WordNetLemmatizer instance.

    Returns:
        List of lemmatized tokens.
    """
    return [lemmatizer.lemmatize(tok) for tok in tokens]


def preprocess_pipeline(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Run the full per-row NLP preprocessing pipeline on a single text value:
    clean -> tokenize -> remove stopwords -> lemmatize -> rejoin into a string.

    Args:
        text: Raw input text.
        stop_words: Set of stopwords to remove.
        lemmatizer: An initialized WordNetLemmatizer instance.

    Returns:
        Fully preprocessed text as a single space-joined string.
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens, stop_words)
    tokens = lemmatize_tokens(tokens, lemmatizer)
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def save_dataset(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        filepath: Destination CSV path.
    """
    df.to_csv(filepath, index=False)
    logger.info(f"Saved cleaned dataset to {filepath} ({len(df)} rows)")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_pipeline(
    input_path: str,
    output_path: str,
    text_column: str = "text",
    cleaned_column: str = "cleaned_text",
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline end-to-end:
    load -> remove duplicates -> handle missing values ->
    clean/tokenize/remove stopwords/lemmatize -> save.

    Args:
        input_path: Path to the raw input CSV file.
        output_path: Path to write the cleaned output CSV file.
        text_column: Name of the column containing raw text.
        cleaned_column: Name of the column to store the preprocessed text.

    Returns:
        The final cleaned DataFrame (also written to output_path).
    """
    download_nltk_resources()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df = load_dataset(input_path)
    df = remove_duplicates(df, subset=text_column)
    df = handle_missing_values(df, subset=text_column)

    logger.info("Applying text cleaning, tokenization, stopword removal, and lemmatization...")
    df[cleaned_column] = df[text_column].apply(
        lambda t: preprocess_pipeline(t, stop_words, lemmatizer)
    )

    # Drop any rows that became empty after cleaning (e.g. text was only URLs/punctuation)
    before = len(df)
    df = df[df[cleaned_column].str.strip() != ""].reset_index(drop=True)
    if before - len(df) > 0:
        logger.info(f"Removed {before - len(df)} rows that were empty after cleaning")

    save_dataset(df, output_path)
    return df


if __name__ == "__main__":
    run_pipeline(
        input_path="dataset.csv",
        output_path="dataset_cleaned.csv",
        text_column="text",
    )
