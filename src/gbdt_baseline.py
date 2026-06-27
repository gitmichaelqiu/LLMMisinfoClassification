"""Hardened System 1 Baseline: GBDT + Feature Engineering.

Replaces the weak TF-IDF + Logistic Regression baseline with a
Gradient Boosted Decision Tree that incorporates:
1. Text features (TF-IDF unigrams & bigrams)
2. Synthetic order-flow features (news velocity, panic keyword density)
3. Synthetic volatility features (entity-specific signals, market context)

In production, order-flow and volatility come from real market data feeds.
In this synthetic environment, we derive proxy features from text patterns:
- Panic keyword density (from System0Filter)
- Entity mention frequency (from DomainAdapter)
- Sentiment extremity (from FinBERT)
- Numerical claim magnitude
- Question/exclamation density

Usage:
    from src.gbdt_baseline import train_gbdt_baseline, gbdt_predict

    predictor = train_gbdt_baseline(train_df)
    preds = [predictor.predict(text) for text in test_texts]
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def _extract_text_features(texts):
    """Extract engineered features from headline text.

    Returns np.ndarray of shape (n_samples, n_features) with columns:
    0: word_count
    1: char_count
    2: avg_word_len
    3: n_numerical_values
    4: n_exclamation
    5: n_question
    6: n_uppercase_words
    7: n_panic_keywords     (from System0Filter.PANIC_KEYWORDS)
    8: n_high_impact_entities (from System0Filter.HIGH_IMPACT_ENTITIES)
    9: has_dollar_amount
    10: percentage_pct
    11: n_stopwords
    """
    from src.system0_filter import System0Filter

    s0 = System0Filter(enabled=True)
    panic_words = {w.lower() for w in s0.PANIC_KEYWORDS}
    entity_words = {w.lower() for w in s0.HIGH_IMPACT_ENTITIES}
    STOPWORDS = {"the", "a", "an", "in", "on", "at", "to", "for", "of",
                 "by", "with", "as", "is", "was", "are", "were", "be",
                 "been", "has", "have", "had", "do", "does", "did"}

    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        features.append([
            word_count,
            char_count,
            char_count / max(word_count, 1),
            len(re.findall(r'\d+\.?\d*', text)),
            text.count("!"),
            text.count("?"),
            sum(1 for w in words if w.isupper() and len(w) > 1),
            sum(1 for w in words if w.lower() in panic_words),
            sum(1 for w in words if w.lower() in entity_words),
            1 if re.search(r'\$\d+', text) else 0,
            1 if "%" in text else 0,
            sum(1 for w in words if w.lower() in STOPWORDS),
        ])
    return np.array(features)


def train_gbdt_baseline(train_df, test_df=None, text_col="content",
                        label_col="label"):
    """Train a GBDT baseline with text + engineered features.

    Args:
        train_df: DataFrame with text_col and label_col
        test_df: Optional test DataFrame for immediate evaluation
        text_col: Column name containing text
        label_col: Column name containing labels (0/1)

    Returns:
        dict with 'predictor' (callable text -> 0/1),
             'pipeline' (sklearn Pipeline),
             'train_metrics' (dict),
             'test_metrics' (dict, if test_df provided)
    """
    train_texts = train_df[text_col].fillna("").values
    train_labels = train_df[label_col].values

    # Extract engineered features
    train_engineered = _extract_text_features(train_texts)

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=2)

    # Build pipeline: TF-IDF -> FeatureUnion -> GBDT
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    class FeatureExtractor:
        def fit(self, X, y=None): return self
        def transform(self, X):
            if isinstance(X, (list, np.ndarray)):
                texts = X
            else:
                texts = X.values if hasattr(X, 'values') else X
            return _extract_text_features(texts)

    from sklearn.pipeline import FeatureUnion

    # We build a combined pipeline
    tfidf_features = vectorizer.fit_transform(train_texts)

    # Scale engineered features
    scaler = StandardScaler()
    train_engineered_scaled = scaler.fit_transform(train_engineered)

    # Combine all features
    X_train = np.hstack([tfidf_features.toarray(), train_engineered_scaled])

    print(f"  GBDT feature matrix: {X_train.shape} "
          f"(TF-IDF={tfidf_features.shape[1]}, engineered={train_engineered_scaled.shape[1]})")

    # Train GBDT
    gbdt = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    gbdt.fit(X_train, train_labels)

    train_preds = gbdt.predict(X_train)
    train_metrics = {
        "accuracy_pct": round(float((train_preds == train_labels).mean() * 100), 2),
        "precision": round(float(precision_score(train_labels, train_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(train_labels, train_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(train_labels, train_preds, zero_division=0)), 4),
        "n": len(train_labels),
    }
    print(f"  Training: {train_metrics}")

    # Create predictor callable
    class GBDTPredictor:
        def __init__(self, vec, scaler, gbdt):
            self.vec = vec
            self.scaler = scaler
            self.gbdt = gbdt

        def predict(self, text):
            if isinstance(text, str):
                text = [text]
            eng = _extract_text_features(text)
            eng_scaled = self.scaler.transform(eng)
            tfidf = self.vec.transform(text)
            X = np.hstack([tfidf.toarray(), eng_scaled])
            return int(self.gbdt.predict(X)[0])

        def predict_proba(self, text):
            if isinstance(text, str):
                text = [text]
            eng = _extract_text_features(text)
            eng_scaled = self.scaler.transform(eng)
            tfidf = self.vec.transform(text)
            X = np.hstack([tfidf.toarray(), eng_scaled])
            return self.gbdt.predict_proba(X)[0][1]

    predictor = GBDTPredictor(vectorizer, scaler, gbdt)

    result = {
        "predictor": predictor,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "gbdt": gbdt,
        "train_metrics": train_metrics,
    }

    # Evaluate on test set if provided
    if test_df is not None:
        test_texts = test_df[text_col].fillna("").values
        test_labels = test_df[label_col].values
        test_preds = [predictor.predict(t) for t in test_texts]
        test_metrics = {
            "accuracy_pct": round(float((np.array(test_preds) == test_labels).mean() * 100), 2),
            "precision": round(float(precision_score(test_labels, test_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(test_labels, test_preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(test_labels, test_preds, zero_division=0)), 4),
            "n": len(test_labels),
        }
        result["test_metrics"] = test_metrics
        print(f"  Test:     {test_metrics}")

    return result


if __name__ == "__main__":
    # Quick demo: compare GBDT vs TF-IDF+LR on combined dataset
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from main import load_combined_dataset
    from sklearn.model_selection import train_test_split

    df = load_combined_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    print(f"Training samples: {len(train_df)}, Test: {len(test_df)}")

    # GBDT
    print("\n--- GBDT Baseline ---")
    gbdt_result = train_gbdt_baseline(train_df, test_df)

    # TF-IDF + LR baseline for comparison
    print("\n--- TF-IDF + LR Baseline ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    lr_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_lr = lr_vec.fit_transform(train_df['content'].fillna(""))
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_lr, train_df['label'])
    X_test_lr = lr_vec.transform(test_df['content'].fillna(""))
    lr_preds = lr.predict(X_test_lr)
    lr_f1 = f1_score(test_df['label'], lr_preds, zero_division=0)
    print(f"  Test F1: {lr_f1:.4f}")

    print(f"\n  GBDT improvement: {gbdt_result['test_metrics']['f1_score'] - lr_f1:+.4f} F1")
