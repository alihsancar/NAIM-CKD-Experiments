
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# CSV dosyalarının bulunduğu klasör
INPUT_DIR = Path("outputs_ckd_modular")

# PNG tablolarının kaydedileceği klasör
OUTPUT_DIR = Path("png_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def csv_to_png(csv_path):
    # CSV dosyasını oku
    df = pd.read_csv(csv_path)

    # Sütun isimlerini daha okunaklı yap
    df.columns = [
        str(col).replace("_", " ").title()
        for col in df.columns
    ]

    # Tablo boyutunu satır ve sütun sayısına göre ayarla
    fig_width = max(10, len(df.columns) * 1.7)
    fig_height = max(2.5, (len(df) + 1) * 0.45)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # Başlık
    title = csv_path.stem.replace("_", " ").title()
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)

    # Tabloyu oluştur
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Başlık satırını ve hücre kenarlıklarını biçimlendir
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)

        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.2)

    # PNG olarak kaydet
    output_path = OUTPUT_DIR / f"{csv_path.stem}.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.2,
    )
    plt.close(fig)

    print(f"Kaydedildi: {output_path}")


# Klasördeki tüm CSV dosyalarını dönüştür
csv_files = list(INPUT_DIR.glob("*.csv"))

if not csv_files:
    print(f"CSV bulunamadı: {INPUT_DIR.resolve()}")
else:
    for csv_file in csv_files:
        try:
            csv_to_png(csv_file)
        except Exception as error:
            print(f"Hata ({csv_file.name}): {error}")

    print("\nDönüştürme işlemi tamamlandı.")
