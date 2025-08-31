import pandas as pd
import numpy as np

import re

import kagglehub
import json

from underthesea import text_normalize as vn_normalize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




path = kagglehub.dataset_download("haitranquangofficial/vietnamese-online-news-dataset")
json_path = path + "/news_dataset.json"
print("Path to dataset file:", json_path)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
data[:5]

raw_data = pd.read_json(json_path)
raw_data.head()


# Preprocessing
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", "", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

VN_STOPWORDS = set("""
và hoặc nhưng là thì mà của những được bằng với về trong trên dưới từ tới cho các một một số
có không chưa đã đang sẽ nữa nhé ạ à ừ ơ dạ vâng ơi nhỉ hả cái này kia ấy vậy thôi luôn
""".split())

def tokenize_vi(text: str) -> str:
    if not text:
        return ""

    # Text normalization
    text = vn_normalize(text)

    # Text cleaning
    text = clean_text(text)

    # Tokenize
    toks = word_tokenize(text, format="text").split()

    # Remove stopwords and digits
    toks = [t for t in toks if t not in VN_STOPWORDS and not t.isdigit()]
    try:
        lemmas = []
        for pair in lemmatize(" ".join(toks)):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                lemmas.append(pair[1])
            else:
                lemmas.append(str(pair))
        toks = lemmas
    except Exception:
        pass
    return toks

def process_text(text: str) -> str:
    return " ".join(tokenize_vi(text))

# Test
text = raw_data.iloc[0]['content']
print(text)
print(process_text(text))

# Remove missing values and duplicates
raw_data = raw_data.dropna()
raw_data = raw_data[~raw_data.duplicated()]

# Sampling
sample_data = raw_data.sample(n=1000, random_state=42)
sample_data.head()

sample_data["processed_text"] = sample_data["content"].apply(process_text)
print(sample_data[["content", "processed_text"]].head())

processed_data = sample_data["processed_text"]
processed_data

cv_uni  = CountVectorizer(ngram_range=(1,1), min_df=2, max_df=0.9)
cv_bi   = CountVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)   # BoW bigram
tf_uni  = TfidfVectorizer(ngram_range=(1,1), min_df=2, max_df=0.9, norm="l2")
tf_bi   = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, norm="l2")

X_bow_uni = cv_uni.fit_transform(processed_data)
X_bow_bi  = cv_bi.fit_transform(processed_data)
X_tfidf_uni = tf_uni.fit_transform(processed_data)
X_tfidf_bi  = tf_bi.fit_transform(processed_data)

# Result
def corpus_stats(raw, proc):
    import numpy as np
    raw_len = raw.str.len()
    tok_counts = proc.str.split().apply(len)
    return pd.DataFrame({
        "num_docs":[len(raw)],
        "avg_raw_chars":[raw_len.mean()],
        "avg_tokens":[tok_counts.mean()],
        "median_tokens":[tok_counts.median()],
        "min_tokens":[tok_counts.min()],
        "max_tokens":[tok_counts.max()],
    })

stats_df = corpus_stats(processed_data, processed_data)
print(stats_df.round(2))

def matrix_report(X, name):
    nnz = X.nnz
    shape = X.shape
    sparsity = 1 - nnz/(shape[0]*shape[1])
    return pd.Series({
        "representation": name,
        "num_docs": shape[0],
        "vocab_size": shape[1],
        "nonzeros": nnz,
        "sparsity": round(sparsity, 4)
    })

rep_df = pd.concat([
    matrix_report(X_bow_uni,  "BoW unigram"),
    matrix_report(X_bow_bi,   "BoW uni+bi"),
    matrix_report(X_tfidf_uni,"TF-IDF unigram"),
    matrix_report(X_tfidf_bi, "TF-IDF uni+bi"),
], axis=1).T
print(rep_df)

def top_terms_count(cv, X, k=20):
    vocab = np.array(cv.get_feature_names_out())
    freqs = np.asarray(X.sum(axis=0)).ravel()
    idx = freqs.argsort()[::-1][:k]
    return pd.DataFrame({"term": vocab[idx], "freq": freqs[idx].astype(int)})

def top_terms_tfidf(tfv, X, k=20):
    vocab = np.array(tfv.get_feature_names_out())
    max_w = X.max(axis=0).toarray().ravel()
    idx = max_w.argsort()[::-1][:k]
    return pd.DataFrame({"term": vocab[idx], "max_tfidf": np.round(max_w[idx], 4)})

top_bow_uni = top_terms_count(cv_uni, X_bow_uni, 20)
top_bow_bi  = top_terms_count(cv_bi,  X_bow_bi,  20)
top_tf_uni  = top_terms_tfidf(tf_uni, X_tfidf_uni, 20)
top_tf_bi   = top_terms_tfidf(tf_bi,  X_tfidf_bi,  20)

for name, df in [("BoW unigram", top_bow_uni),
                 ("BoW uni+bi",  top_bow_bi),
                 ("TF-IDF unigram", top_tf_uni),
                 ("TF-IDF uni+bi",  top_tf_bi)]:
    print("\n===", name, "===")
    print(df.to_string(index=False))

