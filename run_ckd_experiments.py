from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from torch.utils.data import DataLoader, TensorDataset

from CMC_utils.models.tabular.naim.naim import NAIM


# --------------------------------------------------
# 1. Genel ayarlar
# --------------------------------------------------

SEED_LIST = [42, 123, 2026]
MISSING_RATES = [0.10, 0.30, 0.50]

DATA_PATH = Path("datasets/tabular/classification/ckd/ckd.csv")
OUTPUT_DIR = Path("outputs_ckd")

TARGET = "class"

NUMERIC_COLUMNS = [
    "age", "bp", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wbcc", "rbcc",
]

CATEGORICAL_COLUMNS = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

CLASSES = ["ckd", "notckd"]

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. Veri setini yükleme
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri seti bulunamadı: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    # Hedef etiketi eksik olan satırları kullanmıyoruz.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # Sayısal sütunları sayısal tipe dönüştür.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Kategorik sütunları tutarlı biçimde sakla.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].apply(
            lambda value: str(value).strip()
            if pd.notna(value) else np.nan
        )

    # Sınıfları 0 ve 1 olarak kodla.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET].astype(str))

    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()

    print("Veri seti boyutu:", X.shape)
    print("Sınıflar:", list(label_encoder.classes_))
    print("Doğal eksik hücre sayısı:", int(X.isna().sum().sum()))

    return X, y, label_encoder


# --------------------------------------------------
# 3. Ek MCAR eksiklik maskesi oluşturma
# --------------------------------------------------

def apply_missing_mask(X, rate, seed):
    """
    Doğal eksikleri korur ve gözlenen hücrelerin belirtilen
    oranını rastgele eksik hâle getirir.
    """
    X_masked = X.copy()
    rng = np.random.default_rng(seed)

    observed_positions = np.argwhere(
        X.notna().to_numpy()
    )

    n_to_mask = int(np.floor(rate * len(observed_positions)))

    if n_to_mask > 0:
        selected_indices = rng.choice(
            len(observed_positions),
            size=n_to_mask,
            replace=False,
        )

        selected_positions = observed_positions[selected_indices]

        for row_idx, col_idx in selected_positions:
            X_masked.iat[row_idx, col_idx] = np.nan

    return X_masked, n_to_mask


# --------------------------------------------------
# 4. Klasik modellerin hazırlanması
# --------------------------------------------------

def build_classical_models(seed):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])

    models = {
        "SVM": Pipeline([
            ("preprocessor", copy.deepcopy(preprocessor)),
            ("classifier", SVC(
                class_weight="balanced",
                random_state=seed,
            )),
        ]),

        "KNN": Pipeline([
            ("preprocessor", copy.deepcopy(preprocessor)),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
            )),
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", copy.deepcopy(preprocessor)),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }

    return models


# --------------------------------------------------
# 5. Performans metrikleri
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "recall_ckd": recall_score(
            y_true,
            y_pred,
            labels=[0],
            average=None,
            zero_division=0,
        )[0],

        "recall_notckd": recall_score(
            y_true,
            y_pred,
            labels=[1],
            average=None,
            zero_division=0,
        )[0],
    }

# --------------------------------------------------
# 6. NAIM için veri hazırlama
# --------------------------------------------------

def prepare_naim_data(X_train, X_val, X_test):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    train_arrays = []
    val_arrays = []
    test_arrays = []

    cat_idxs = []
    cat_dims = []

    for col in NUMERIC_COLUMNS:
        train_col = X_train[col].to_numpy(dtype=np.float32)
        val_col = X_val[col].to_numpy(dtype=np.float32)
        test_col = X_test[col].to_numpy(dtype=np.float32)

        # Ölçekleme parametreleri yalnızca eğitim verisinden öğrenilir.
        min_value = np.nanmin(train_col)
        max_value = np.nanmax(train_col)
        scale = max_value - min_value

        if scale > 0:
            train_col = (train_col - min_value) / scale
            val_col = (val_col - min_value) / scale
            test_col = (test_col - min_value) / scale
        else:
            train_col = np.where(np.isnan(train_col), np.nan, 0.0)
            val_col = np.where(np.isnan(val_col), np.nan, 0.0)
            test_col = np.where(np.isnan(test_col), np.nan, 0.0)

        train_arrays.append(train_col)
        val_arrays.append(val_col)
        test_arrays.append(test_col)

    for col in CATEGORICAL_COLUMNS:
        # Kategorik sözlük sadece eğitim verisinden oluşturulur.
        categories = sorted(X_train[col].dropna().unique().tolist())
        category_to_index = {
            category: idx for idx, category in enumerate(categories)
        }

        # NAIM, eksik kategorik değerleri cat_dims ile temsil eder.
        missing_category_index = len(categories)

        def encode_category(series):
            return np.array([
                category_to_index.get(value, np.nan)
                if pd.notna(value) else np.nan
                for value in series
            ], dtype=np.float32)

        train_arrays.append(encode_category(X_train[col]))
        val_arrays.append(encode_category(X_val[col]))
        test_arrays.append(encode_category(X_test[col]))

        cat_idxs.append(len(train_arrays) - 1)
        cat_dims.append(missing_category_index)

    X_train_array = np.column_stack(train_arrays).astype(np.float32)
    X_val_array = np.column_stack(val_arrays).astype(np.float32)
    X_test_array = np.column_stack(test_arrays).astype(np.float32)

    return (
        X_train_array,
        X_val_array,
        X_test_array,
        cat_idxs,
        cat_dims,
    )


# --------------------------------------------------
# 7. NAIM eğitimi ve tahmini
# --------------------------------------------------

def train_and_predict_naim(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    seed,
    max_epochs=50,
    patience=10,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    (
        X_train_array,
        X_val_array,
        X_test_array,
        cat_idxs,
        cat_dims,
    ) = prepare_naim_data(X_train, X_val, X_test)

    model = NAIM(
        input_size=X_train_array.shape[1],
        output_size=2,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        d_token=6,
        embedder_initialization="normal",
        bias=False,
        mask_type=2,
        missing_value="~inf",
        num_heads=3,
        feedforward_dim=128,
        dropout_rate=0.1,
        activation="relu",
        num_layers=2,
        extractor=False,
    ).to(DEVICE)

    counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (
        2 * np.maximum(counts, 1)
    )
    weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=weights_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_array, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=generator,
    )

    X_val_tensor = torch.tensor(
        X_val_array,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_val_array = np.asarray(y_val)

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    best_train_f1 = 0.0
    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            # Eğitim verisindeki tahminler
            train_logits = model(X_train_tensor)
            train_predictions = train_logits.argmax(dim=1).cpu().numpy()

            # Doğrulama verisindeki tahminler
            val_logits = model(X_val_tensor)
            val_predictions = val_logits.argmax(dim=1).cpu().numpy()

        train_f1 = f1_score(
            np.asarray(y_train),
            train_predictions,
            average="macro",
            zero_division=0,
        )

        val_f1 = f1_score(
            y_val_array,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_train_f1 = train_f1
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("NAIM için en iyi model durumu kaydedilemedi.")

    model.load_state_dict(best_state)
    model.eval()

    X_test_tensor = torch.tensor(
        X_test_array,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        test_logits = model(X_test_tensor)
        predictions = test_logits.argmax(dim=1).cpu().numpy()

    return predictions

# --------------------------------------------------
# 8. Deneyleri çalıştırma
# --------------------------------------------------

def run_experiments():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, label_encoder = load_data()
    all_results = []
    mask_audit = []

    for seed in SEED_LIST:
        print(f"\n{'=' * 60}")
        print(f"SEED: {seed}")
        print(f"{'=' * 60}")

        # Önce %70 eğitim, %30 geçici veri.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=seed,
            stratify=y,
        )

        # Geçici veriyi eşit bölerek %15 doğrulama, %15 test.
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=seed,
            stratify=y_temp,
        )

        for rate in MISSING_RATES:
            print(f"\nEksiklik oranı: %{rate * 100:.0f}")

            # Her veri bölümü için maskeyi yalnızca bir kez üret.
            X_train_masked, n_train_masked = apply_missing_mask(
                X_train, rate, seed + 1000
            )
            X_val_masked, n_val_masked = apply_missing_mask(
                X_val, rate, seed + 2000
            )
            X_test_masked, n_test_masked = apply_missing_mask(
                X_test, rate, seed + 3000
            )

            mask_audit.append({
                "seed": seed,
                "missing_rate": rate,
                "train_cells_masked": n_train_masked,
                "val_cells_masked": n_val_masked,
                "test_cells_masked": n_test_masked,
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
            })

            # Klasik modeller: imputasyon her modelin eğitiminde öğrenilir.
            classical_models = build_classical_models(seed)

            for model_name, model in classical_models.items():
                print(f"  {model_name} eğitiliyor...")

                try:
                    model.fit(X_train_masked, y_train)
                    predictions = model.predict(X_test_masked)

                    metrics = calculate_metrics(y_test, predictions)

                    all_results.append({
                        "seed": seed,
                        "missing_rate": rate,
                        "model": model_name,
                        **metrics,
                        "status": "ok",
                        "error": "",
                    })

                    print(
                        f"    Accuracy={metrics['accuracy']:.3f}, "
                        f"Macro-F1={metrics['macro_f1']:.3f}"
                    )

                except Exception as exc:
                    print(f"    HATA: {exc}")

                    all_results.append({
                        "seed": seed,
                        "missing_rate": rate,
                        "model": model_name,
                        "accuracy": np.nan,
                        "macro_f1": np.nan,
                        "recall_ckd": np.nan,
                        "recall_notckd": np.nan,
                        "status": "error",
                        "error": str(exc),
                    })

                pd.DataFrame(all_results).to_csv(
                    OUTPUT_DIR / "results_by_seed.csv",
                    index=False,
                )

            # NAIM: imputasyon uygulamadan eğit.
            print("  NAIM eğitiliyor...")

            try:
                predictions = train_and_predict_naim(
                    X_train_masked,
                    y_train,
                    X_val_masked,
                    y_val,
                    X_test_masked,
                    seed=seed,
                )

                metrics = calculate_metrics(y_test, predictions)

                all_results.append({
                    "seed": seed,
                    "missing_rate": rate,
                    "model": "NAIM",
                    **metrics,
                    "status": "ok",
                    "error": "",
                })

                print(
                    f"    Accuracy={metrics['accuracy']:.3f}, "
                    f"Macro-F1={metrics['macro_f1']:.3f}"
                )

            except Exception as exc:
                print(f"    NAIM HATASI: {exc}")

                all_results.append({
                    "seed": seed,
                    "missing_rate": rate,
                    "model": "NAIM",
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "recall_ckd": np.nan,
                    "recall_notckd": np.nan,
                    "status": "error",
                    "error": str(exc),
                })

            # Her kombinasyondan sonra sonuçları kaydet.
            pd.DataFrame(all_results).to_csv(
                OUTPUT_DIR / "results_by_seed.csv",
                index=False,
            )

            pd.DataFrame(mask_audit).to_csv(
                OUTPUT_DIR / "mask_audit.csv",
                index=False,
            )

    # --------------------------------------------------
    # 9. Ortalama ve standart sapma
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        OUTPUT_DIR / "results_by_seed.csv",
        index=False,
    )

    successful = results_df[results_df["status"] == "ok"].copy()

    if successful.empty:
        print("\nBaşarılı deney sonucu yok. Hataları kontrol et.")
        return

    metrics_to_summarize = [
        "accuracy",
        "macro_f1",
        "recall_ckd",
        "recall_notckd",
    ]

    summary = (
        successful
        .groupby(["model", "missing_rate"])[metrics_to_summarize]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    summary.to_csv(
        OUTPUT_DIR / "summary_mean_std.csv",
        index=False,
    )

    # --------------------------------------------------
    # 10. Grafikler
    # --------------------------------------------------

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in metrics_to_summarize:
        plt.figure(figsize=(9, 5))

        for model_name in successful["model"].unique():
            model_results = (
                successful[successful["model"] == model_name]
                .groupby("missing_rate")[metric]
                .agg(["mean", "std"])
                .sort_index()
            )

            plt.errorbar(
                model_results.index * 100,
                model_results["mean"],
                yerr=model_results["std"].fillna(0),
                marker="o",
                capsize=4,
                label=model_name,
            )

        plt.xlabel("Ek MCAR eksiklik oranı (%)")
        plt.ylabel(metric)
        plt.title(f"{metric} - Model karşılaştırması")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{metric}.png",
            dpi=150,
        )
        plt.close()

    print("\nDeneyler tamamlandı.")
    print("Sonuç klasörü:", OUTPUT_DIR.resolve())
    print("Başarılı deney sayısı:", len(successful))
    print("Hatalı deney sayısı:", len(results_df) - len(successful))


if __name__ == "__main__":
    run_experiments()