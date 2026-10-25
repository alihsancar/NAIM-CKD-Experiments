
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from config import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def build_classical_models(seed):
    """
    Create SVM, KNN, and Random Forest pipelines.

    Imputation and preprocessing are fitted only on training data
    when the pipeline's fit() method is called.
    """

    # Numeric features:
    # 1. Fill missing values with the training median.
    # 2. Standardize the features.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical features:
    # 1. Fill missing values with the training mode.
    # 2. Convert categories into one-hot encoded features.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Apply the appropriate preprocessing to each column group.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    models = {
        "SVM": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    SVC(
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),

        "KNN": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    KNeighborsClassifier(n_neighbors=5),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models
