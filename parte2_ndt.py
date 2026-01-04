import pandas as pd
import numpy as np
from scipy.stats import chi2

df = pd.read_csv("ndt_tests_tratado.csv")
mle = pd.read_excel("mle_output/mle_summary.xlsx")

cliente_A = "client10"
cliente_B = "client13"

df_A = df[df["client"] == cliente_A]
df_B = df[df["client"] == cliente_B]

if df_A.empty or df_B.empty:
    raise ValueError(
        f"Erro: dados vazios para {cliente_A} ou {cliente_B}. "
        "Verifique o nome dos clientes no CSV."
    )


# 6.1 – LRT para Throughput (Gamma–Gamma)
Y_A = (
    df_A["download_throughput_bps"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .astype(float)
    .values
)

Y_B = (
    df_B["download_throughput_bps"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .astype(float)
    .values
)

if len(Y_A) == 0 or len(Y_B) == 0:
    raise ValueError("Erro: throughput vazio após dropna().")

nA = len(Y_A)
nB = len(Y_B)

Ybar_A = Y_A.mean()
Ybar_B = Y_B.mean()
Y = np.concatenate([Y_A, Y_B])
Ybar = Y.mean()

# k_MLE estimado por MLE clássico
k_mle = (Y.mean() ** 2) / Y.var(ddof=0)

W_throughput = 2 * k_mle * (
    nA * np.log(Ybar / Ybar_A) +
    nB * np.log(Ybar / Ybar_B)
)

alpha = 0.05
chi_crit = chi2.ppf(1 - alpha, df=1)

print("=== LRT Throughput (Gamma–Gamma) ===")
print("k_MLE:", k_mle)
print("W observado:", W_throughput)
print("Qui-quadrado crítico:", chi_crit)

if W_throughput > chi_crit:
    print("Rejeita H0: throughputs médios diferentes")
else:
    print("Não rejeita H0")


# 6.2 – LRT para RTT (Normal–Normal)
mu_hat, sigma_hat = eval(
    mle.loc[mle["variável"] == "rtt_download_sec", "θ̂_MLE"].values[0]
)

sigma2_mle = sigma_hat ** 2

R_A = (
    df_A["rtt_download_sec"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .astype(float)
    .values
)

R_B = (
    df_B["rtt_download_sec"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .astype(float)
    .values
)

if len(R_A) == 0 or len(R_B) == 0:
    raise ValueError("Erro: RTT vazio após dropna().")

nA = len(R_A)
nB = len(R_B)

Rbar_A = R_A.mean()
Rbar_B = R_B.mean()

W_rtt = (1 / sigma2_mle) * (nA * nB / (nA + nB)) * (Rbar_A - Rbar_B)**2

chi_crit = chi2.ppf(1 - alpha, df=1)

print("\n=== LRT RTT (Normal–Normal) ===")
print("σ²_MLE:", sigma2_mle)
print("W observado:", W_rtt)
print("Qui-quadrado crítico:", chi_crit)

if W_rtt > chi_crit:
    print("Rejeita H0: RTTs médios diferentes")
else:
    print("Não rejeita H0")