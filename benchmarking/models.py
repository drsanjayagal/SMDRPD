"""Estimator registry. XGBoost and SHAP are imported lazily and are optional."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SUPPORTED_MODELS = (
    "logistic_regression",
    "svm",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "mlp",
)


def build_model(name: str, seed: int = 0, n_classes: int = 3):
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
    if name == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        max_iter=200,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed; pip install xgboost") from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            objective="multi:softprob",
            num_class=n_classes,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")