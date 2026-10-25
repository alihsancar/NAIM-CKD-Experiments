
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Dosya yolları
DATA_PATH = Path("datasets/tabular/classification/ckd/ckd.csv")
OUTPUT_DIR = Path("outputs_ckd_modular/data_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Veri kümesini oku; '?' işaretlerini eksik değer say
df = pd.read_csv(DATA_PATH, na_values=["?"])

target_col = "class"

numeric_features = [
    "age", "bp", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wbcc", "rbcc"
]

categorical_features = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

# 1. Genel veri kümesi özeti
n_rows = len(df)
n_features = len(df.columns) - 1
feature_df = df.drop(columns=[target_col])

overview = pd.DataFrame({
    "Ölçüt": [
        "Gözlem sayısı",
        "Özellik sayısı",
        "Özelliklerdeki toplam hücre sayısı",
        "Eksik özellik hücresi sayısı",
        "Eksik özellik hücrelerinin oranı (%)"
    ],
    "Değer": [
        n_rows,
        n_features,
        feature_df.size,
        int(feature_df.isna().sum().sum()),
        round(feature_df.isna().sum().sum() / feature_df.size * 100, 2)
    ]
})

overview.to_csv(
    OUTPUT_DIR / "dataset_overview.csv",
    index=False,
    encoding="utf-8-sig"
)

# 2. Değişken türleri ve eksik değer analizi
rows = []

for col in df.columns:
    if col == target_col:
        continue

    if col in numeric_features:
        variable_type = "Sayısal"
    elif col in categorical_features:
        variable_type = "Kategorik"
    else:
        variable_type = "Kontrol edilmeli"

    missing_count = int(df[col].isna().sum())

    rows.append({
        "Değişken": col,
        "Tür": variable_type,
        "Eksik değer sayısı": missing_count,
        "Eksik değer oranı (%)": round(
            missing_count / len(df) * 100, 2
        ),
        "Gözlenen değer sayısı": int(df[col].notna().sum())
    })

variable_summary = pd.DataFrame(rows)
variable_summary.to_csv(
    OUTPUT_DIR / "variable_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# 3. Sınıf dağılımı
class_counts = df[target_col].value_counts(dropna=False)
class_summary = pd.DataFrame({
    "Sınıf": class_counts.index.astype(str),
    "Gözlem sayısı": class_counts.values,
    "Oran (%)": (
        class_counts.values / len(df) * 100
    ).round(2)
})

class_summary.to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

# Sınıf dağılımı PNG
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(
    class_summary["Sınıf"],
    class_summary["Gözlem sayısı"]
)
ax.set_title("CKD Veri Kümesi - Sınıf Dağılımı")
ax.set_xlabel("Sınıf")
ax.set_ylabel("Gözlem sayısı")

for i, value in enumerate(class_summary["Gözlem sayısı"]):
    ax.text(i, value, str(value), ha="center", va="bottom")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Değişken başına eksik değer oranı PNG
plot_df = variable_summary.sort_values(
    "Eksik değer oranı (%)",
    ascending=True
)

fig_height = max(6, len(plot_df) * 0.3)
fig, ax = plt.subplots(figsize=(9, fig_height))

ax.barh(
    plot_df["Değişken"],
    plot_df["Eksik değer oranı (%)"]
)
ax.set_title("Değişkenlere Göre Doğal Eksik Veri Oranları")
ax.set_xlabel("Eksik değer oranı (%)")
ax.set_ylabel("Değişken")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "missingness_by_variable.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Veri analizi tamamlandı.")
print(f"Çıktı klasörü: {OUTPUT_DIR.resolve()}")
print("\nVeri kümesi özeti:")
print(overview.to_string(index=False))
print("\nSınıf dağılımı:")
print(class_summary.to_string(index=False))
print("\nDeğişken analizi:")
print(variable_summary.to_string(index=False))
