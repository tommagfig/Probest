"""
EDA for M-Lab NDT subset (ndt_tests_tratado.csv)
------------------------------------------------
Steps:
1) Load the CSV (expects ndt_tests_tratado.csv in the same folder).
2) Use fixed column mapping for the known dataset schema.
3) Compute descriptive stats by CLIENT and by SERVER:
   - count, mean, median, variance, std, min, max, q90, q95, q99
4) Save summary tables to Excel (eda_summary.xlsx).
5) Auto-select two entities (clients or servers) with contrasting behavior.
6) Create histograms, boxplots, and scatter plots (RTT vs Throughput).
Outputs are saved to ./eda_output/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Config ---------------------------
INPUT_CSV = "ndt_tests_tratado.csv"
OUTPUT_DIR = "eda_output"
QUANTILES = [0.5, 0.9, 0.95, 0.99]

# --------------------------- Fixed Mapping --------------------
# Baseado no formato do seu CSV
mapping = {
    "timestamp": "timestamp",
    "client": "client",
    "server": "server",
    "thr_down": "download_throughput_bps",
    "thr_up": "upload_throughput_bps",
    "rtt_down": "rtt_download_sec",
    "rtt_up": "rtt_upload_sec",
    "loss_pct": "packet_loss_percent",
}

# ---------------------- Utility Functions ---------------------
def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def summarize_by_group(df, group_col, var_cols):
    """Compute descriptive stats grouped by group_col."""
    stats = {}
    for var in var_cols:
        if var not in df.columns:
            continue
        series = safe_to_numeric(df[var])
        if not np.issubdtype(series.dropna().dtype, np.number):
            continue

        desc = df.groupby(group_col)[var].agg(["count", "mean", "median", "var", "std", "min", "max"])
        qdf = df.groupby(group_col)[var].quantile(QUANTILES).unstack(level=-1)
        qdf.columns = [f"q{int(q*100)}" if q != 0.5 else "q50" for q in qdf.columns]
        stats[var] = desc.join(qdf, how="left")

    if not stats:
        return pd.DataFrame()
    out = pd.concat(stats, axis=1)
    return out.sort_index()

def plot_hist_box_scatter(df, entity_col, entity_value, mapping, tag):
    """Create histograms, boxplots and scatter plots for one entity."""
    subset = df[df[entity_col] == entity_value].copy()
    for key in ["thr_up", "thr_down", "rtt_up", "rtt_down", "loss_pct"]:
        col = mapping[key]
        if col in subset.columns:
            subset[col] = safe_to_numeric(subset[col])

    out_dir = os.path.join(OUTPUT_DIR, f"{tag}_{entity_value}")
    ensure_dir(out_dir)

    var_pairs = [
        (mapping["thr_down"], "Throughput (Down) [bps]"),
        (mapping["thr_up"], "Throughput (Up) [bps]"),
        (mapping["rtt_down"], "RTT (Down) [s]"),
        (mapping["rtt_up"], "RTT (Up) [s]"),
        (mapping["loss_pct"], "Perda de Pacotes [%]"),
    ]

    # Hist e boxplot
    for col, label in var_pairs:
        if col not in subset.columns:
            continue
        plt.figure(figsize=(7, 5))
        subset[col].dropna().plot(kind="hist", bins=40, edgecolor="black")
        plt.title(f"{tag}={entity_value} — Histograma {label}")
        plt.xlabel(label)
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{col}.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.boxplot(subset[col].dropna(), vert=True, labels=[label])
        plt.title(f"{tag}={entity_value} — Boxplot {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_{col}.png"), dpi=160)
        plt.close()

    # Scatter RTT vs Throughput
    for thr, rtt, label in [
        (mapping["thr_down"], mapping["rtt_down"], "Down"),
        (mapping["thr_up"], mapping["rtt_up"], "Up"),
    ]:
        if thr in subset.columns and rtt in subset.columns:
            x = safe_to_numeric(subset[rtt])
            y = safe_to_numeric(subset[thr])
            ok = x.notna() & y.notna()
            if ok.any():
                plt.figure(figsize=(7, 5))
                plt.scatter(x[ok], y[ok], alpha=0.6, s=12)
                plt.xlabel(f"RTT ({label}) [s]")
                plt.ylabel(f"Throughput ({label}) [bps]")
                plt.title(f"{tag}={entity_value} — Scatter RTT {label} vs Throughput {label}")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"scatter_{label}.png"), dpi=160)
                plt.close()

def auto_pick_entities(df):
    """Pick two interesting clients: highest q99 RTT_down and lowest median RTT_down."""
    tag_key, tag_col = "client", mapping["client"]
    rd = mapping["rtt_down"]

    temp = df[[tag_col, rd]].copy()
    temp[rd] = pd.to_numeric(temp[rd], errors="coerce")
    gp = temp.groupby(tag_col)[rd]
    q99 = gp.quantile(0.99).sort_values(ascending=False)
    med = gp.median().sort_values(ascending=True)
    if len(q99) >= 1 and len(med) >= 1:
        pick1 = q99.index[0]
        pick2 = med.index[0] if med.index[0] != pick1 else med.index[1]
        return tag_key, tag_col, [pick1, pick2]
    return None

# ------------------------------- MAIN -------------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"ERROR: '{INPUT_CSV}' not found. Put it in this folder and run again.")

    print("Reading CSV efficiently...")
    dtypes = {
        mapping["thr_down"]: "float64",
        mapping["thr_up"]: "float64",
        mapping["rtt_down"]: "float64",
        mapping["rtt_up"]: "float64",
        mapping["loss_pct"]: "float64",
        mapping["client"]: "category",
        mapping["server"]: "category",
    }
    print("Reading CSV safely...")

    def clean_number(x):
        if isinstance(x, str):
            x = x.replace('.', '').replace(',', '.')
        return x

    converters = {
        mapping["thr_down"]: clean_number,
        mapping["thr_up"]: clean_number,
        mapping["rtt_down"]: clean_number,
        mapping["rtt_up"]: clean_number,
        mapping["loss_pct"]: clean_number,
    }

    df = pd.read_csv(
        INPUT_CSV,
        converters=converters,
        parse_dates=[mapping["timestamp"]],
        low_memory=False
    )

    for col in [mapping["thr_down"], mapping["thr_up"], mapping["rtt_down"], mapping["rtt_up"], mapping["loss_pct"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    print("Computing statistics...")
    vars_ = [mapping[k] for k in ["thr_up", "thr_down", "rtt_up", "rtt_down", "loss_pct"]]

    ensure_dir(OUTPUT_DIR)
    excel_path = os.path.join(OUTPUT_DIR, "eda_summary.xlsx")

    by_client = summarize_by_group(df, mapping["client"], vars_)
    by_server = summarize_by_group(df, mapping["server"], vars_)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        by_client.to_excel(writer, sheet_name="by_client")
        by_server.to_excel(writer, sheet_name="by_server")
        pd.DataFrame.from_dict(mapping, orient="index", columns=["mapped_column"]).to_excel(writer, sheet_name="column_mapping")

    print("Selecting two contrasting clients for plots...")
    pick = auto_pick_entities(df)
    selections_info = []
    if pick:
        tag_key, tag_col, picks = pick
        for p in picks[:2]:
            plot_hist_box_scatter(df, tag_col, p, mapping, tag=tag_key)
        selections_info = [f"{tag_key}={p}" for p in picks[:2]]

    md = [
        "# EDA — NDT Subset",
        "",
        "## Quantis escolhidos",
        "- **q90** e **q99** são relevantes para desempenho de rede porque capturam *tail latency* (as maiores latências que impactam a experiência).",
        "- **q95** dá um panorama intermediário entre mediana e cauda extrema.",
        "",
        "## Arquivos gerados",
        f"- Tabelas: `{excel_path}`",
    ]
    if selections_info:
        md.append(f"- Gráficos: {', '.join(selections_info)} (pasta `eda_output/`)")
    md += [
        "",
        "## Observações",
        "- Compare os resumos por cliente e por servidor (planilhas `by_client` e `by_server`).",
        "- Observe as diferenças em média, variância e quantis — especialmente q99 de RTT.",
        "- Nos *scatter plots*, verifique a correlação entre RTT e Throughput (espera-se relação negativa em muitos cenários).",
    ]
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n✅ EDA completa!")
    print(f"Excel salvo em: {excel_path}")
    if selections_info:
        print("Gráficos gerados para:", ", ".join(selections_info))
    print("Saída em:", OUTPUT_DIR)

if __name__ == "__main__":
    main()