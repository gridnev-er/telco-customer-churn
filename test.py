import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

# 1) Мини‑табличка
X = pd.DataFrame({"a": [0, 1, 2, 3], "text": ["hi", "hi hi", "hello", "goodbye"]})
y = np.array([0, 0, 1, 1])

# 2) Простейший текстовый пайплайн
pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200))
proba = pipe.fit(X["text"], y).predict_proba(X["text"])[:, 1]
print("ROC-AUC (только текст):", roc_auc_score(y, proba))

# 3) Проверка Parquet (нужен pyarrow)
df = pd.DataFrame({"x": [1, 2, 3]})
df.to_parquet("/tmp/check.parquet")
print(pd.read_parquet("/tmp/check.parquet").shape)
