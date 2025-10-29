import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ===================== Config =====================
INPUT_CSV = "ndt_tests_tratado.csv"
OUTPUT_DIR = "bayes_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Escolha de entidade para analisar (use 'client'/'server' + valor)
ENTITY_COL = "client"      # "client" ou "server"
ENTITY_ID  = "client10"    # ex.: "client10", "server07", etc.

# Split train/test
TRAIN_FRAC = 0.70
N_PACKETS  = 1000  # para Beta–Binomial (perda)

# Priors (ajuste se quiser)
# Normal–Normal (μ | σ² conhecida): μ ~ Normal(μ0, τ0²)
MU0   = 0.0
TAU02 = 1e6  # variância grande => prior fracamente informativa

# Gamma–Gamma (β): β ~ Gamma(a0, b0)   (parametrização: shape=a, rate=b)
A0_GAM = 1.0
B0_GAM = 1e-6

# Beta–Binomial: p ~ Beta(a0, b0)
A0_BETA = 1.0
B0_BETA = 1.0

# ===================== Helpers =====================
def clean_numeric_col(s):
    # opcional: se seu CSV tiver separador de milhar
    if s.dtype == object:
        s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors="coerce")

def time_split(df, frac=0.7, ts_col="timestamp", seed=42):
    df = df.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df.sort_values(ts_col)
        n = len(df)
        cut = int(np.floor(frac * n))
        return df.iloc[:cut], df.iloc[cut:]
    # fallback (sem timestamp)
    return df.sample(frac=frac, random_state=seed), df.drop(df.sample(frac=frac, random_state=seed).index)

def summarize_pred_vs_test(name, pred_mean, pred_var, test_series):
    test_series = pd.to_numeric(test_series, errors="coerce").dropna()
    return {
        "variavel": name,
        "E_pred": float(pred_mean),
        "Var_pred": float(pred_var),
        "Media_teste": float(test_series.mean()) if len(test_series) else np.nan,
        "Var_teste": float(test_series.var(ddof=1)) if len(test_series) > 1 else np.nan,
        "n_teste": int(len(test_series)),
    }

# ===================== Load & prepare =====================
df = pd.read_csv(INPUT_CSV)

cols = [
    "download_throughput_bps",
    "upload_throughput_bps",
    "rtt_download_sec",
    "rtt_upload_sec",
    "packet_loss_percent",
    "timestamp",
    ENTITY_COL,
]
cols = [c for c in cols if c in df.columns]
df = df[cols].copy()

# limpar números
for c in df.columns:
    if c in ["timestamp", ENTITY_COL]:
        continue
    df[c] = clean_numeric_col(df[c])

# filtra entidade
df_ent = df[df[ENTITY_COL] == ENTITY_ID].copy()
if df_ent.empty:
    raise SystemExit(f"Nenhuma linha para {ENTITY_COL}={ENTITY_ID}")

train, test = time_split(df_ent, frac=TRAIN_FRAC, ts_col="timestamp")

# ===================== 1) RTT: Normal–Normal =====================
results_rows = []

for rtt_col in ["rtt_download_sec", "rtt_upload_sec"]:
    if rtt_col not in train.columns:
        continue
    y_tr = train[rtt_col].dropna().values
    y_te = test[rtt_col].dropna().values
    if len(y_tr) == 0:
        continue

    # Likelihood: ri | μ ~ N(μ, σ²), com σ² conhecido = s2_hat do treino (MLE)
    s2_hat = float(np.var(y_tr, ddof=1)) if len(y_tr) > 1 else float(np.var(y_tr, ddof=0))
    n = len(y_tr)
    ybar = float(np.mean(y_tr))

    # Posterior: μ | r ~ Normal(μn, τn²)
    # τn² = (1/τ0² + n/σ²)^(-1)
    tau_n2 = 1.0 / (1.0/TAU02 + n/s2_hat) if s2_hat > 0 else TAU02
    mu_n   = tau_n2 * (MU0/TAU02 + n*ybar/s2_hat) if s2_hat > 0 else ybar

    # Predictive: Ynew ~ N(μn, σ² + τn²)
    pred_mean = mu_n
    pred_var  = s2_hat + tau_n2

    # Salva comparação com teste
    results_rows.append(summarize_pred_vs_test(f"{rtt_col} (Normal)", pred_mean, pred_var, test[rtt_col]))

    # Gráfico: hist teste + densidade preditiva
    if len(y_te) > 0:
        xs = np.linspace(min(y_te), max(y_te), 300)
        pdf = stats.norm.pdf(xs, loc=mu_n, scale=np.sqrt(pred_var))
        plt.figure(figsize=(7,5))
        plt.hist(y_te, bins=40, density=True, alpha=0.6, edgecolor="black")
        plt.plot(xs, pdf, lw=2, label="Posterior Predictive (Normal)")
        plt.title(f"{ENTITY_COL}={ENTITY_ID} — {rtt_col} — preditiva vs teste")
        plt.xlabel(rtt_col); plt.ylabel("densidade")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{rtt_col}_predictive.png"), dpi=150)
        plt.close()

# ===================== 2) Throughput: Gamma–Gamma (β) =====================
# Parametrização: y | β ~ Gamma(k, β)    (shape=k, rate=β)
# Prior: β ~ Gamma(a0, b0)
# Posterior: β | y ~ Gamma(a_n, b_n),  a_n = a0 + n*k,  b_n = b0 + sum(y)
# Predictive: Ynew tem Beta-prime escalada; E[Ynew] = k * b_n / (a_n - 1) se a_n > 1
# Var[Ynew] = k (k + a_n - 1) b_n^2 / ((a_n - 1)^2 (a_n - 2))  se a_n > 2

def fit_gamma_mle_shape(y):
    # usa scipy para shape MLE k̂ com loc=0; retorna k̂ e rate β̂ (1/scale)
    a_hat, loc_hat, scale_hat = stats.gamma.fit(y, floc=0)
    k_hat = float(a_hat)
    beta_hat = float(1.0 / scale_hat)
    return k_hat, beta_hat

for thr_col in ["download_throughput_bps", "upload_throughput_bps"]:
    if thr_col not in train.columns:
        continue
    y_tr = train[thr_col].dropna().values
    y_te = test[thr_col].dropna().values
    if len(y_tr) == 0:
        continue

    # k̂ (shape) por MLE no TREINO
    k_hat, beta_hat = fit_gamma_mle_shape(y_tr)
    n = len(y_tr)
    sum_y = float(np.sum(y_tr))

    # Posterior de β
    a_n = A0_GAM + n * k_hat
    b_n = B0_GAM + sum_y

    # Preditiva: média/variância (se existir)
    if a_n > 1:
        pred_mean = k_hat * (b_n / (a_n - 1))
    else:
        pred_mean = np.nan
    if a_n > 2:
        pred_var = (k_hat * (k_hat + a_n - 1) * (b_n**2)) / ((a_n - 1)**2 * (a_n - 2))
    else:
        pred_var = np.nan

    results_rows.append(summarize_pred_vs_test(f"{thr_col} (Gamma-Gamma, k̂ fixo={k_hat:.3f})",
                                               pred_mean, pred_var, test[thr_col]))

    # Gráfico: hist teste + densidade preditiva (aproximação via MC)
    if len(y_te) > 0:
        # Amostragem: β ~ Gamma(a_n, b_n), depois Y ~ Gamma(k_hat, β)
        mcs = 20000
        beta_s = stats.gamma.rvs(a=a_n, scale=1.0/b_n, size=mcs)  # note scale = 1/rate
        y_pred = stats.gamma.rvs(a=k_hat, scale=1.0/beta_s)
        xs = np.linspace(min(y_te), max(y_te), 300)
        # Suaviza densidade preditiva com KDE
        kde = stats.gaussian_kde(y_pred)
        pdf = kde(xs)

        plt.figure(figsize=(7,5))
        plt.hist(y_te, bins=40, density=True, alpha=0.6, edgecolor="black")
        plt.plot(xs, pdf, lw=2, label="Posterior Predictive (MC)")
        plt.title(f"{ENTITY_COL}={ENTITY_ID} — {thr_col} — preditiva vs teste")
        plt.xlabel(thr_col); plt.ylabel("densidade")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{thr_col}_predictive.png"), dpi=150)
        plt.close()

# ===================== 3) Perda: Beta–Binomial =====================
# Converte percentuais em proporção e em contagem com N_PACKETS
loss_col = "packet_loss_percent"
if loss_col in train.columns:
    p_tr = (train[loss_col] / 100.0).clip(0, 1).dropna().values
    p_te = (test[loss_col]  / 100.0).clip(0, 1).dropna().values

    # cria contagens
    x_tr = np.clip(np.round(p_tr * N_PACKETS), 0, N_PACKETS).astype(int)
    n_tr = np.full_like(x_tr, N_PACKETS)

    # agrega treinos
    x_tot = int(np.sum(x_tr))
    n_tot = int(np.sum(n_tr))

    # Posterior de p: Beta(a_n, b_n)
    a_n = A0_BETA + x_tot
    b_n = B0_BETA + (n_tot - x_tot)

    # Preditiva para fração (E[X/n*], Var[X/n*]) usando n* = N_PACKETS
    n_star = N_PACKETS
    # Contagem:
    mean_count = n_star * (a_n / (a_n + b_n))
    var_count  = (n_star * (a_n * b_n * (a_n + b_n + n_star))) / (((a_n + b_n)**2) * (a_n + b_n + 1))
    # Fração:
    pred_mean_frac = mean_count / n_star
    pred_var_frac  = var_count / (n_star**2)

    # Comparar com teste (fração observada)
    results_rows.append(summarize_pred_vs_test(f"{loss_col} (Beta-Binomial, n={N_PACKETS})",
                                               pred_mean_frac, pred_var_frac, test[loss_col]/100.0))

    # Gráfico: hist fração (teste) + densidade Beta posterior
    if len(p_te) > 0:
        xs = np.linspace(0, min(1, max(0.001, p_te.max()*1.1)), 400)
        beta_pdf = stats.beta.pdf(xs, a=a_n, b=b_n)
        plt.figure(figsize=(7,5))
        plt.hist(p_te, bins=40, density=True, alpha=0.6, edgecolor="black")
        plt.plot(xs, beta_pdf, lw=2, label=f"Posterior Beta(a={a_n}, b={b_n})")
        plt.title(f"{ENTITY_COL}={ENTITY_ID} — {loss_col} — preditiva vs teste (fração)")
        plt.xlabel("loss fraction"); plt.ylabel("densidade")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{loss_col}_predictive.png"), dpi=150)
        plt.close()

# ===================== Exporta resumo =====================
summary_df = pd.DataFrame(results_rows)
summary_path = os.path.join(OUTPUT_DIR, f"bayes_summary_{ENTITY_COL}_{ENTITY_ID}.xlsx")
summary_df.to_excel(summary_path, index=False)

print("\n✅ Bayes 3.3 concluído!")
print(f"Resumo em: {summary_path}")
print(f"Gráficos preditivos salvos em: {OUTPUT_DIR}/")