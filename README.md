# Week 1 Text Processing Report

This report summarizes the contents and workflow implemented in `week1.ipynb`.

## Overview

The notebook focuses on **text preprocessing and vectorization** steps for Natural Language Processing (NLP) tasks.

It covers:

- Text cleaning (removing URLs, emails, mentions, special characters).

- Text normalization (Vietnamese-specific using underthesea).

- Tokenization, stopword removal, and lemmatization.

- Vectorization using Bag-of-Words and TF-IDF, with unigram and bigram features.

- Statistical reporting of dataset before and after preprocessing.

## How to Run

1. Install dependencies:

   ```bash
   pip install underthesea scikit-learn nbconvert
   ```

2. Open the notebook in Jupyter:

   ```bash
   jupyter notebook week1.ipynb
   ```

3. Or run the exported script:

   ```bash
   python week1.py
   ```

## Outputs

- Preprocessed dataset column `processed_text`.

- Sparse matrices: `X_bow_uni`, `X_bow_bi`, `X_tfidf_uni`, `X_tfidf_bi`.

- Corpus statistics (average tokens, min/max tokens, sparsity).

- Top terms identified by Bag-of-Words and TF-IDF.

## Notes

- Only **1,000 sample rows** were used to save computational resources and speed up the experiment.  

- Some rows may become empty after preprocessing (min_tokens = 0). These can be dropped or replaced with a special `<empty>` token.

- Adjust `min_df` and `max_df` in vectorizers depending on dataset size and vocabulary distribution.
