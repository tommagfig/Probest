
"""
EDA for M-Lab NDT subset (ndt_tests_tratado.csv)
------------------------------------------------
What this script does:
1) Load the CSV (expects ndt_tests_tratado.csv in the same folder).
2) Heuristically map column names for: timestamp, client, server,
   throughput_up, throughput_down, rtt_up, rtt_down, loss_pct.
3) Compute descriptive stats by CLIENT and by SERVER:
   - count, mean, median, variance, std, min, max, q90, q95, q99
4) Save summary tables to Excel (eda_summary.xlsx).
5) Auto-select two entities (clients or client+server) with contrasting behavior.
6) Create histograms, boxplots, and scatter plots (RTT vs Throughput) for the selections.
Outputs are saved to ./eda_output/
"""

import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Config ---------------------------
INPUT_CSV = "ndt_tests_tratado.csv"
OUTPUT_DIR = "eda_output"
QUANTILES = [0.5, 0.9, 0.95, 0.99]

# --------------------- Helper: Column mapping -----------------
def find_first_col(df, patterns):
    """
    Return first column name that matches any of the regex patterns (case-insensitive).
    """
    for col in df.columns:
        for pat in patterns:
            if re.search(pat, col, flags=re.IGNORECASE):
                return col
    return None

def infer_columns(df):
    """
    Try to infer column names from typical Portuguese/English terms.
    Returns a dict with keys:
    timestamp, client, server, thr_up, thr_down, rtt_up, rtt_down, loss_pct
    """
    cols = list(df.columns)

    # Timestamp: assume first column OR look for date/time related names
    timestamp = cols[0]
    ts_guess = find_first_col(df, [r"data", r"date", r"timestamp", r"hora", r"time"])
    if ts_guess is not None:
        timestamp = ts_guess

    # Client / Server
    client = find_first_col(df, [r"cliente", r"client", r"source.*id", r"host.*id"])
    server = find_first_col(df, [r"servidor", r"server", r"dest.*id"])

    # Throughput Down / Up (bps)
    thr_down = find_first_col(df, [r"throughput.*down", r"download.*bps", r"down.*bps", r"down.*throughput", r"download.*bit"])
    thr_up = find_first_col(df, [r"throughput.*up", r"upload.*bps", r"up.*bps", r"up.*throughput", r"upload.*bit"])

    # RTT Down / Up (seconds)
    rtt_down = find_first_col(df, [r"rtt.*down", r"download.*rtt", r"rtt.*download"])
    rtt_up = find_first_col(df, [r"rtt.*up", r"upload.*rtt", r"rtt.*upload"])

    # Loss percentage
    loss_pct = find_first_col(df, [r"loss", r"perda", r"packet.*loss", r"perdas"])

    mapping = dict(timestamp=timestamp, client=client, server=server,
                   thr_up=thr_up, thr_down=thr_down, rtt_up=rtt_up, rtt_down=rtt_down, loss_pct=loss_pct)

    return mapping

def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def summarize_by_group(df, group_col, var_cols):
    """
    For each var in var_cols, compute descriptive stats grouped by group_col.
    Returns a MultiIndex columns DataFrame.
    """
    stats = {}
    for var in var_cols:
        if var is None or var not in df.columns:
            continue
        series = df[var]
        # Only include numeric
        if not np.issubdtype(series.dropna().dtype, np.number):
            continue

        desc = df.groupby(group_col)[var].agg(["count", "mean", "median", "var", "std", "min", "max"])
        # quantiles
        qdf = df.groupby(group_col)[var].quantile(QUANTILES).unstack(level=-1)
        # nicer quantile column names
        qdf.columns = [f"q{int(q*100)}" if q != 0.5 else "q50" for q in qdf.columns]
        stats[var] = desc.join(qdf, how="left")

    # Combine into one big table
    if not stats:
        return pd.DataFrame()

    out = pd.concat(stats, axis=1)
    # sort index by natural order
    out = out.sort_index()
    return out

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def plot_hist_box_scatter(df, entity_col, entity_value, mapping, tag):
    """
    For a given entity (client or server), create plots:
    - Histograms for thr_up, thr_down, rtt_up, rtt_down, loss_pct
    - Boxplots for the same
    - Scatter: RTT_down vs Throughput_down; RTT_up vs Throughput_up
    """
    subset = df[df[entity_col] == entity_value].copy()
    # Cast numerics
    for key in ["thr_up", "thr_down", "rtt_up", "rtt_down", "loss_pct"]:
        col = mapping.get(key)
        if col and col in subset.columns:
            subset[col] = safe_to_numeric(subset[col])

    out_dir = os.path.join(OUTPUT_DIR, f"{tag}_{entity_value}")
    ensure_dir(out_dir)

    # Variables to plot if present
    var_pairs = [
        (mapping.get("thr_down"), "Throughput (Down) [bps]"),
        (mapping.get("thr_up"), "Throughput (Up) [bps]"),
        (mapping.get("rtt_down"), "RTT (Down) [s]"),
        (mapping.get("rtt_up"), "RTT (Up) [s]"),
        (mapping.get("loss_pct"), "Perda de Pacotes [%]")
    ]
    numeric_cols = [(c, label) for c, label in var_pairs if c and c in subset.columns and np.issubdtype(subset[c].dropna().dtype, np.number)]

    # Histograms and boxplots
    for col, label in numeric_cols:
        # Histogram
        plt.figure(figsize=(7,5))
        subset[col].dropna().plot(kind="hist", bins=40, edgecolor="black")
        plt.title(f"{tag}={entity_value} — Histograma {label}")
        plt.xlabel(label)
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{col}.png"), dpi=160)
        plt.close()

        # Boxplot
        plt.figure(figsize=(6,5))
        plt.boxplot(subset[col].dropna(), vert=True, labels=[label])
        plt.title(f"{tag}={entity_value} — Boxplot {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_{col}.png"), dpi=160)
        plt.close()

    # Scatter plots (RTT vs Throughput) for up and down
    # Down
    td, rd = mapping.get("thr_down"), mapping.get("rtt_down")
    if td and rd and td in subset.columns and rd in subset.columns:
        x = safe_to_numeric(subset[rd])
        y = safe_to_numeric(subset[td])
        ok = x.notna() & y.notna()
        if ok.any():
            plt.figure(figsize=(7,5))
            plt.scatter(x[ok], y[ok], alpha=0.6, s=12)
            plt.xlabel("RTT (Down) [s]")
            plt.ylabel("Throughput (Down) [bps]")
            plt.title(f"{tag}={entity_value} — Scatter RTT Down vs Throughput Down")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"scatter_rttDown_vs_thrDown.png"), dpi=160)
            plt.close()

    # Up
    tu, ru = mapping.get("thr_up"), mapping.get("rtt_up")
    if tu and ru and tu in subset.columns and ru in subset.columns:
        x = safe_to_numeric(subset[ru])
        y = safe_to_numeric(subset[tu])
        ok = x.notna() & y.notna()
        if ok.any():
            plt.figure(figsize=(7,5))
            plt.scatter(x[ok], y[ok], alpha=0.6, s=12)
            plt.xlabel("RTT (Up) [s]")
            plt.ylabel("Throughput (Up) [bps]")
            plt.title(f"{tag}={entity_value} — Scatter RTT Up vs Throughput Up")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"scatter_rttUp_vs_thrUp.png"), dpi=160)
            plt.close()

def auto_pick_entities(df, mapping):
    """
    Pick two interesting entities:
    1) Client with highest q99 RTT_down (if available)
    2) Client with lowest median RTT_down (contrast)
    If client information missing, fallback to servers.
    """
    # choose which entity type to use
    entity_type = None
    if mapping.get("client") in df.columns:
        entity_type = ("client", mapping["client"])
    elif mapping.get("server") in df.columns:
        entity_type = ("server", mapping["server"])
    else:
        return None

    tag_key, tag_col = entity_type

    rd = mapping.get("rtt_down")
    if rd and rd in df.columns:
        temp = df[[tag_col, rd]].copy()
        temp[rd] = pd.to_numeric(temp[rd], errors="coerce")
        gp = temp.groupby(tag_col)[rd]
        q99 = gp.quantile(0.99).sort_values(ascending=False)
        med = gp.median().sort_values(ascending=True)
        if len(q99) >= 1 and len(med) >= 1:
            pick1 = q99.index[0]
            # ensure different entities if possible
            pick2 = med.index[0] if med.index[0] != pick1 else (med.index[1] if len(med) > 1 else med.index[0])
            return (tag_key, tag_col, [pick1, pick2])

    # fallback: use throughput_down variance
    td = mapping.get("thr_down")
    if td and td in df.columns:
        temp = df[[tag_col, td]].copy()
        temp[td] = pd.to_numeric(temp[td], errors="coerce")
        var = temp.groupby(tag_col)[td].var().sort_values(ascending=False)
        if len(var) >= 2:
            return (tag_key, tag_col, list(var.index[:2]))

    return None

def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"ERROR: '{INPUT_CSV}' not found. Put it in this folder and run again.")

    df = pd.read_csv(INPUT_CSV)
    mapping = infer_columns(df)

    # Parse timestamp if possible
    ts_col = mapping.get("timestamp")
    if ts_col and ts_col in df.columns:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        except Exception:
            pass

    # Select numeric columns of interest that exist
    num_keys = ["thr_up", "thr_down", "rtt_up", "rtt_down", "loss_pct"]
    present_vars = [mapping[k] for k in num_keys if mapping.get(k) in df.columns]

    # By CLIENT
    client_col = mapping.get("client")
    by_client = pd.DataFrame()
    if client_col and client_col in df.columns:
        by_client = summarize_by_group(df, client_col, present_vars)

    # By SERVER
    server_col = mapping.get("server")
    by_server = pd.DataFrame()
    if server_col and server_col in df.columns:
        by_server = summarize_by_group(df, server_col, present_vars)

    # Save Excel with both tables
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    excel_path = os.path.join(OUTPUT_DIR, "eda_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        if not by_client.empty:
            by_client.to_excel(writer, sheet_name="by_client")
        if not by_server.empty:
            by_server.to_excel(writer, sheet_name="by_server")
        # also save mapping sheet
        pd.DataFrame.from_dict(mapping, orient="index", columns=["mapped_column"]).to_excel(writer, sheet_name="column_mapping")

    # Auto-pick two entities
    pick = auto_pick_entities(df, mapping)
    selections_info = []
    if pick is not None:
        tag_key, tag_col, picks = pick
        for p in picks[:2]:
            plot_hist_box_scatter(df, tag_col, p, mapping, tag=tag_key)
        selections_info = [f"{tag_key}={p}" for p in picks[:2]]

    # Save a small markdown summary
    md = []
    md.append("# EDA — NDT Subset")
    md.append("")
    md.append("## Quantis escolhidos")
    md.append("- **q90** e **q99** são relevantes para desempenho de rede porque capturam *tail latency* (as maiores latências que impactam a experiência).")
    md.append("- **q95** dá um panorama intermediário entre mediana e cauda extrema.")
    md.append("")
    md.append("## Arquivos gerados")
    md.append(f"- Tabelas: `{excel_path}`")
    if selections_info:
        md.append(f"- Gráficos para: {', '.join(selections_info)} (pasta `eda_output/{tag_key}_<id>/`)")
    md.append("")
    md.append("## Observações")
    md.append("- Compare os resumos por cliente e por servidor (planilhas `by_client` e `by_server`).")
    md.append("- Observe as diferenças em média, variância e quantis—especialmente q99 de RTT.")
    md.append("- Nos *scatter plots*, verifique a correlação entre RTT e Throughput (espera-se relação negativa em muitos cenários).")
    md_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("Done.")
    print(f"Column mapping guess: {mapping}")
    if selections_info:
        print("Auto-selected for plots:", ", ".join(selections_info))
    print(f"Outputs in: {OUTPUT_DIR}")
    print(f"- {excel_path}")
    print(f"- {md_path}")

if __name__ == "__main__":
    main()
