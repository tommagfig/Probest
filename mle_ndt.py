import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# ------------------ Configuração ------------------
INPUT_CSV = "ndt_tests_tratado.csv"
OUTPUT_DIR = "mle_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ Carregar dados ------------------
df = pd.read_csv(INPUT_CSV)

cols = [
    "download_throughput_bps",
    "upload_throughput_bps",
    "rtt_download_sec",
    "rtt_upload_sec",
    "packet_loss_percent"
]

for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------------ Escolher um cliente exemplo ------------------
subset = df[df["client"] == "client10"]

# ------------------ Modelos escolhidos ------------------
models = [
    ("download_throughput_bps", stats.gamma, "Gamma"),
    ("upload_throughput_bps", stats.gamma, "Gamma"),
    ("rtt_download_sec", stats.norm, "Normal"),
    ("rtt_upload_sec", stats.norm, "Normal"),
    ("packet_loss_percent", stats.beta, "Beta")
]

results = []

# ------------------ Ajuste por MLE ------------------
for col, dist, dist_name in models:
    data = subset[col].dropna()
    if len(data) == 0:
        continue

    # Ajustes e pré-processamentos específicos
    if dist_name == "Beta":
        data = data / 100  # converter percentual para proporção
        # Evitar zeros e uns exatos (que travam o fit da Beta)
        data = np.clip(data, 1e-6, 1 - 1e-6)

    try:
        params = dist.fit(data, floc=0 if dist_name != "Normal" else None)
        results.append((col, dist_name, params))
        print(f"{col} ({dist_name}) — parâmetros MLE:", params)

        # ---------- Histograma + PDF ajustada ----------
        plt.figure(figsize=(7,5))
        plt.hist(data, bins=40, density=True, alpha=0.6, color='gray', edgecolor='black')
        x = np.linspace(min(data), max(data), 200)
        plt.plot(x, dist.pdf(x, *params), 'r-', lw=2, label=f"{dist_name} (MLE)")
        plt.title(f"{col} — Ajuste {dist_name} (client10)")
        plt.xlabel(col)
        plt.ylabel("Densidade")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_mle_fit.png"), dpi=150)
        plt.close()

        # ---------- QQ Plot ----------
        plt.figure(figsize=(6,5))
        stats.probplot(data, dist=dist, sparams=params, plot=plt)
        plt.title(f"QQ-Plot — {col} ({dist_name}, MLE)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_qqplot.png"), dpi=150)
        plt.close()

    except Exception as e:
        print(f"Falha ao ajustar {col}: {e}")

# ------------------ Exportar parâmetros MLE ------------------
df_results = pd.DataFrame(results, columns=["variável", "modelo", "θ̂_MLE"])
df_results.to_excel(os.path.join(OUTPUT_DIR, "mle_summary.xlsx"), index=False)

print("\n✅ Ajustes concluídos!")
print(f"Resultados salvos em: {OUTPUT_DIR}/mle_summary.xlsx")
print("Gráficos: histogramas e QQ-plots salvos em mle_output/")