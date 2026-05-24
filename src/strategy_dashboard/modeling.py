from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def _topic_label(terms: list[str]) -> str:
    joined = " ".join(terms)
    if any(word in joined for word in ["alarm", "alert", "low", "night"]):
        return "Safety and alert reliability"
    if any(word in joined for word in ["insurance", "cost", "pay", "expensive"]):
        return "Affordability and access"
    if any(word in joined for word in ["app", "bluetooth", "connection", "phone"]):
        return "App and connectivity friction"
    if any(word in joined for word in ["pump", "loop", "share", "caregiver"]):
        return "Integration ecosystem"
    if any(word in joined for word in ["skin", "adhesive", "wear", "pain"]):
        return "Wearability and comfort"
    return "Accuracy and daily usability"


def derive_topics(frame: pd.DataFrame, *, n_topics: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    texts = frame["clean_text"].fillna("").astype(str)
    n_topics = max(1, min(n_topics, len(frame), 8))
    vectorizer = TfidfVectorizer(max_features=1500, min_df=1, max_df=0.95, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
    doc_topics = model.fit_transform(matrix).argmax(axis=1)
    terms = vectorizer.get_feature_names_out()

    topic_rows = []
    for index, weights in enumerate(model.components_):
        top_terms = [terms[i] for i in weights.argsort()[-8:][::-1]]
        topic_rows.append({"topic": index, "label": _topic_label(top_terms), "top_terms": ", ".join(top_terms)})

    out = frame.copy()
    out["topic"] = doc_topics
    return out, pd.DataFrame(topic_rows)


def derive_segments(frame: pd.DataFrame, *, n_segments: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    texts = frame["clean_text"].fillna("").astype(str)
    n_segments = max(1, min(n_segments, len(frame), 6))
    vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.95, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=n_segments, n_init=10, random_state=42)
    labels = model.fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()

    rows = []
    for segment in range(n_segments):
        sub = frame.iloc[[i for i, label in enumerate(labels) if label == segment]]
        top_terms = [terms[i] for i in model.cluster_centers_[segment].argsort()[-8:][::-1]]
        rows.append(
            {
                "segment": segment,
                "persona": _topic_label(top_terms),
                "size": len(sub),
                "share": len(sub) / max(len(frame), 1),
                "top_terms": ", ".join(top_terms),
                "dominant_brand": sub["brand"].mode().iat[0] if len(sub) else "Unknown",
                "negative_share": sub["sentiment"].eq("Negative").mean() if len(sub) else 0,
            }
        )

    out = frame.copy()
    out["segment"] = labels
    return out, pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
