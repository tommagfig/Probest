import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ------------------ Carregar dados ------------------
df = pd.read_csv("ndt_tests_tratado.csv")

cols = [
    "download_throughput_bps",
    "upload_throughput_bps",
    "rtt_download_sec",
    "rtt_upload_sec",
    "packet_loss_percent"
]

for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------------ Exemplo para um cliente ------------------
subset = df[df["client"] == "client10"]
data = subset["rtt_download_sec"].dropna()

# Ajustes individuais (exemplo de RTT)
params_gamma = stats.gamma.fit(data, floc=0)
params_lognorm = stats.lognorm.fit(data, floc=0)
params_norm = stats.norm.fit(data)

print("Parâmetros Gamma (MLE):", params_gamma)
print("Parâmetros Lognormal (MLE):", params_lognorm)
print("Parâmetros Normal (MLE):", params_norm)

# ------------------ Histograma + PDFs ------------------
plt.figure(figsize=(7,5))
plt.hist(data, bins=40, density=True, alpha=0.6, color='gray', edgecolor='black')

x = np.linspace(min(data), max(data), 200)
plt.plot(x, stats.gamma.pdf(x, *params_gamma), 'r-', label='Gamma MLE')
plt.plot(x, stats.norm.pdf(x, *params_norm), 'b--', label='Normal MLE')
plt.plot(x, stats.lognorm.pdf(x, *params_lognorm), 'g-.', label='Lognormal MLE')

plt.title("RTT (download) — Ajuste por MLE (client10)")
plt.xlabel("RTT (s)")
plt.ylabel("Densidade")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------ QQ Plot ------------------
stats.probplot(data, dist=stats.gamma, sparams=params_gamma, plot=plt)
plt.title("QQ-Plot — Gamma (MLE)")
plt.show()

# ------------------ Loop geral para todas variáveis ------------------
results = []
for col, dist, dist_name in [
    ("download_throughput_bps", stats.lognorm, "Lognormal"),
    ("upload_throughput_bps", stats.lognorm, "Lognormal"),
    ("rtt_download_sec", stats.gamma, "Gamma"),
    ("rtt_upload_sec", stats.gamma, "Gamma"),
    ("packet_loss_percent", stats.beta, "Beta"),
]:
    data = df[col].dropna()
    if len(data) == 0:
        continue

    if col == "packet_loss_percent":
        data = data / 100  # normalizar para [0,1]

    try:
        params = dist.fit(data, floc=0)
        results.append((col, dist_name, params))

        # Gráfico de ajuste
        plt.figure(figsize=(6,4))
        plt.hist(data, bins=40, density=True, alpha=0.6, color='gray', edgecolor='black')
        x = np.linspace(min(data), max(data), 200)
        plt.plot(x, dist.pdf(x, *params), 'r-', label=f'{dist_name} MLE')
        plt.title(f'{col} — {dist_name} (MLE)')
        plt.xlabel(col)
        plt.ylabel('Densidade')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"mle_{col}.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"Falha ao ajustar {col}: {e}")

pd.DataFrame(results, columns=["variável", "modelo", "θ̂_MLE"]).to_excel("mle_summary.xlsx", index=False)
print("Resultados salvos em mle_summary.xlsx")
