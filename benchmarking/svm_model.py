# benchmarking/svm_model.py
"""Thin wrapper exposing the shared SVM configuration."""
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(seed: int = 0):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf", probability=True, random_state=seed))])