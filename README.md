# IMDB Review Sentiment Classifier

## Introduction
This project builds a terminal-based sentiment analysis application for the IMDB movie review dataset. It follows a standard NLP workflow: data exploration, preprocessing, TF-IDF vectorization, and Multinomial Naive Bayes classification.

## Problem Statement
The goal is to classify each movie review as either positive or negative using textual content only. The model must be transparent, reproducible, and suitable for academic project reporting.

## Objectives
- Load and validate the IMDB dataset.
- Explore the dataset statistically.
- Clean and normalize review text using NLP preprocessing.
- Benchmark the cleaning implementation.
- Train a TF-IDF + Multinomial Naive Bayes model.
- Compare the performance on a balanced sample and on the full dataset.

## Dataset Description
The dataset contains 50,000 IMDB reviews with the following structure:
- `review`: the raw review text.
- `sentiment`: either `positive` or `negative`.

The script automatically locates the CSV file even if the folder name differs slightly from the expected default.

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure
```text
IMDB Sentiment Classifier/
├── Dataset/
├── results/
│   ├── plots/
│   └── output.txt
├── imdb_sentiment_classifier.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

## Phase 1: Data Loading and Exploration
This phase loads the dataset, prints the file shape, checks missing values, validates the key columns, and computes class distribution. It also calculates word-count statistics and reports duplicate rows without silently removing data.

## Phase 2: Cleaning and NLP Preprocessing
The cleaning pipeline removes HTML tags, lowercases the text, tokenizes the review using NLTK, removes stop words while preserving negation words, keeps alphabetic tokens, and finally applies lemmatization. Lemmatisation is preferred over stemming because it produces dictionary-form words that are more semantically accurate for sentiment analysis.

## Phase 3: Classification
The classification step uses TF-IDF features with an n-gram range of `(1, 2)`, `min_df=5`, `max_df=0.8`, `max_features=50000`, and `sublinear_tf=True`. The model is the Multinomial Naive Bayes classifier with `alpha=0.5`.

## Installation Instructions
```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Execution Instructions
```bash
.venv\Scripts\python imdb_sentiment_classifier.py
```

## Experimental Results
The full project was executed with the actual IMDB dataset. The final results are reported in `results/output.txt` and include:
- Dataset shape and class distribution.
- Word-count statistics.
- Duplicate analysis.
- NLP cleaning demonstration.
- Cleaning benchmark.
- Sample model evaluation.
- Full dataset model evaluation.
- Important positive and negative terms.

Actual execution results from the completed run:
- Dataset size: 50,000 reviews, split evenly into 25,000 positive and 25,000 negative reviews.
- Baseline majority accuracy: 0.5000.
- Mean review length: 231.16 words; median: 173.00 words.
- Duplicates detected: 418 rows.
- Efficient cleanup benchmark: 2.462448 seconds vs repeated initialization benchmark: 3.382463 seconds.
- Initial 20,000-review experiment accuracy: 0.8825.
- Full-dataset experiment accuracy: 0.8861.
- Final confusion matrix: [[4385, 615], [524, 4476]].

## Limitations
1. The model depends on the quality of the dataset and does not use contextual embeddings or transformer-based models.
2. It is optimized for binary sentiment classification and does not handle sarcasm or nuanced opinions as strongly as modern deep-learning models.

## Conclusion
The project successfully implements a reproducible NLP sentiment-classification pipeline using TF-IDF and Multinomial Naive Bayes. It is suitable for a college-level machine learning project and demonstrates both the data-preprocessing workflow and model evaluation process.
