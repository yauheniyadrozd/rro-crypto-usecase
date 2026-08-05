"""Fetches Ethereum transactions for a list of wallets, builds a directed
transaction graph, scores each wallet across 7 risk features, writes a
structured report to a text file, and visualises the graph.
"""

import os
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ETHERSCAN_API_KEY = os.environ["ETHERSCAN_API_KEY"]

# Human-readable labels for well-known addresses
KNOWN_WALLETS: dict[str, str] = {
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": "Vitalik Buterin (vitalik.eth)",
    "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B": "Vitalik Buterin (VB2)",
    "0x220866B1A2219f40e72f5c628b65D54268Ca3A9d": "Vitalik Buterin (VB3)",

    # Centralised exchanges
    "0x3f5CE5FBFe3E9af3971dd833D26bA9b5C936f0bE": "Binance (main)",
    "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF": "Binance (hot #2)",
    "0x564286362092D8e7936f0549571a803B203aAceD": "Binance (hot #3)",
    "0xf977814e90dA44bFA03b6295A0616a897441acec": "Binance (hot #20)",
    "0x28C6c06298d514Db089934071355E5743bf21d60": "Binance (#14)",
    "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3": "Coinbase (hot #1)",
    "0xa9D1e08C7793af67e9d92fe308d5697FB81d3E43": "Coinbase (hot #2)",
    "0x77696bb39917C91A0c3908D577d5e322095425cA": "Coinbase (hot #3)",
    "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2": "Kraken (hot #1)",
    "0xAe2D4617c862309A3d75A0fFB358c7a5009c673F": "Kraken (hot #2)",
    "0x6cC5F688a315f3dC28A7781717a9A798a59fDA7b": "OKX (hot)",
    "0x236F233dBf88d2a741E8A4e46b0cF2B3Afa11e68": "OKX (cold #2)",

    # DeFi protocols (contract-controlled, high volume)
    "0x00000000219ab540356cBB839Cbe05303d7705Fa": "ETH 2.0 Beacon Deposit",
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "Wrapped ETH (WETH)",
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": "Uniswap V2 Router",
    "0xE592427A0AEce92De3Edee1F18E0157C05861564": "Uniswap V3 Router",
    "0xDef1C0ded9bec7F1a1670819833240f027b25EfF": "0x Exchange Proxy",

    # Known high-risk / sanctioned / mixer-adjacent
    "0xD4B88Df4D29F5CedD6857912842cff3b20C8Cfa3": "Tornado Cash (100 ETH pool)",
    "0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF": "Tornado Cash (10 ETH pool)",
    "0xbB93e510BbCD0B7beb5A853875f9eC60275CF498": "Tornado Cash (1 ETH pool)",
    "0x8589427373D6D84E98730D7795D8f6f8731FDA16": "Tornado Cash (0.1 ETH pool)",

    # NFT / whale wallets
    "0x1919DB36cA2fa2e15F9000fd9CdC2edcF863E685": "NFT Whale #1 (0x1919)",
}

# Wallets to analyse
WALLETS: list[str] = list(KNOWN_WALLETS.keys())

# General parameters
MIN_VALUE_ETH          = 0.05   # ignore transfers below this threshold (ETH)
MAX_CYCLE_LEN          = 6      # maximum cycle length to detect
DELAY_BETWEEN_REQUESTS = 0.25   # seconds between Etherscan API calls

# Scoring thresholds
TRANSIT_WINDOW_HOURS   = 1      # fast transit: funds in-and-out within N hours
STRUCTURING_THRESHOLD  = 5      # suspicious repeat-transfer count to one address
NIGHT_HOURS            = (2, 6) # UTC hours considered "night activity"
FAN_THRESHOLD          = 10     # fan-in / fan-out node-degree threshold

# Feature weights (must sum to 100)
WEIGHTS: dict[str, int] = {
    "cycle":        30,  # circular transaction pattern
    "transit":      20,  # rapid pass-through of funds
    "structuring":  15,  # splitting transfers to evade detection
    "fan_out":      10,  # high number of unique recipients
    "fan_in":       10,  # high number of unique senders
    "night":        10,  # unusual late-night activity
    "tainted":       5,  # direct link to already-flagged wallets
}

# Output file path
REPORT_PATH = Path("../output/eth_aml_report.txt")


def get_wallet_transactions(address: str) -> pd.DataFrame:
    """Fetch the full normal-transaction history for one wallet via Etherscan v2."""
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": 1,
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99_999_999,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    [!] Request error: {exc}")
        return pd.DataFrame()

    if data["status"] != "1":
        if "No transactions" not in str(data.get("result", "")):
            print(f"    [!] API message: {data.get('message', '?')}")
        return pd.DataFrame()

    df = pd.DataFrame(data["result"])
    if df.empty:
        return df

    df["value_eth"] = df["value"].astype(float) / 10**18
    df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s", utc=True)
    df["from"]      = df["from"].str.lower()
    df["to"]        = df["to"].str.lower()
    df["wallet"]    = address.lower()
    return df[["timeStamp", "from", "to", "value_eth", "hash", "wallet"]]


def fetch_all_wallets(wallets: list[str]) -> pd.DataFrame:
    """Fetch transactions for all watched wallets and deduplicate by tx hash."""
    all_dfs, total = [], len(wallets)
    for i, addr in enumerate(wallets, 1):
        label = KNOWN_WALLETS.get(addr, addr[:14] + "...")
        print(f"  [{i}/{total}] {label}")
        df = get_wallet_transactions(addr)
        if not df.empty:
            all_dfs.append(df)
            print(f"         -> {len(df)} transactions")
        if i < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["hash"])
    return combined

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed weighted graph where edges represent ETH flows."""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        s, r, v = row["from"], row["to"], row["value_eth"]
        if v < MIN_VALUE_ETH or s == r:
            continue
        if G.has_edge(s, r):
            G[s][r]["count"] += 1
            G[s][r]["value"] += v
        else:
            G.add_edge(s, r, count=1, value=v,
                       first_time=row["timeStamp"], tx=row["hash"])
    return G

def _score_cycle(wallet: str, G: nx.DiGraph) -> tuple[float, list[str]]:
    """Score participation in circular transaction loops."""
    cycles = [c for c in nx.simple_cycles(G)
              if wallet in c and len(c) <= MAX_CYCLE_LEN]
    if not cycles:
        return 0.0, []

    score = min(1.0, len(cycles) / 3)
    reasons: list[str] = []
    for c in cycles[:3]:
        path = " -> ".join(_label(n)[:12] for n in c) + f" -> {_label(c[0])[:12]}"
        reasons.append(f"Cycle [{len(c)} nodes]: {path}")
    if len(cycles) > 3:
        reasons.append(f"... and {len(cycles) - 3} more cycle(s)")
    return score, reasons


def _score_transit(wallet: str, df: pd.DataFrame) -> tuple[float, list[str]]:
    """Score rapid transit — funds received and forwarded within a short window."""
    incoming = df[df["to"] == wallet].sort_values("timeStamp")
    outgoing = df[df["from"] == wallet].sort_values("timeStamp")
    if incoming.empty or outgoing.empty:
        return 0.0, []

    transit_count = 0
    for _, inc in incoming.iterrows():
        window_end = inc["timeStamp"] + pd.Timedelta(hours=TRANSIT_WINDOW_HOURS)
        fast_out = outgoing[
            (outgoing["timeStamp"] >= inc["timeStamp"]) &
            (outgoing["timeStamp"] <= window_end)
        ]
        if not fast_out.empty:
            transit_count += 1

    if transit_count == 0:
        return 0.0, []

    ratio = transit_count / max(len(incoming), 1)
    score = min(1.0, ratio)
    return score, [
        f"Fast transit: funds forwarded {transit_count} time(s) within "
        f"{TRANSIT_WINDOW_HOURS}h of receipt ({ratio:.0%} of incoming txs)"
    ]


def _score_structuring(wallet: str, df: pd.DataFrame) -> tuple[float, list[str]]:
    """Score structuring — many small transfers to the same address."""
    sent = df[df["from"] == wallet]
    if sent.empty:
        return 0.0, []

    counts = sent.groupby("to").size()
    suspicious = counts[counts >= STRUCTURING_THRESHOLD]
    if suspicious.empty:
        return 0.0, []

    score = min(1.0, len(suspicious) / 5)
    reasons: list[str] = []
    for addr, cnt in suspicious.head(3).items():
        reasons.append(f"Structuring: {cnt} transfers -> {_label(addr)[:20]}")
    return score, reasons


def _score_fan(
    wallet: str, G: nx.DiGraph
) -> tuple[float, float, list[str], list[str]]:
    """Score fan-out (many unique recipients) and fan-in (many unique senders)."""
    out_deg = G.out_degree(wallet) if wallet in G else 0
    in_deg  = G.in_degree(wallet)  if wallet in G else 0

    score_out = min(1.0, max(0, out_deg - FAN_THRESHOLD) / FAN_THRESHOLD)
    score_in  = min(1.0, max(0, in_deg  - FAN_THRESHOLD) / FAN_THRESHOLD)

    reasons_out = (
        [f"Fan-out: sends to {out_deg} unique addresses (threshold: {FAN_THRESHOLD})"]
        if out_deg >= FAN_THRESHOLD else []
    )
    reasons_in = (
        [f"Fan-in: receives from {in_deg} unique addresses (threshold: {FAN_THRESHOLD})"]
        if in_deg >= FAN_THRESHOLD else []
    )
    return score_out, score_in, reasons_out, reasons_in


def _score_night(wallet: str, df: pd.DataFrame) -> tuple[float, list[str]]:
    """Score suspicious late-night activity (02:00–06:00 UTC)."""
    activity = df[(df["from"] == wallet) | (df["to"] == wallet)]
    if activity.empty:
        return 0.0, []

    night = activity[activity["timeStamp"].dt.hour.between(*NIGHT_HOURS)]
    ratio = len(night) / len(activity)
    if ratio < 0.4:
        return 0.0, []

    score = min(1.0, (ratio - 0.4) / 0.4)
    return score, [
        f"Night activity: {ratio:.0%} of transactions between "
        f"{NIGHT_HOURS[0]:02d}:00 and {NIGHT_HOURS[1]:02d}:00 UTC"
    ]


def _score_tainted(
    wallet: str, G: nx.DiGraph, tainted_set: set
) -> tuple[float, list[str]]:
    """Score direct connections to already-flagged wallets."""
    if wallet not in G:
        return 0.0, []

    neighbors = set(G.successors(wallet)) | set(G.predecessors(wallet))
    tainted_neighbors = neighbors & tainted_set
    if not tainted_neighbors:
        return 0.0, []

    score = min(1.0, len(tainted_neighbors) / 3)
    reasons = [
        f"Linked to flagged wallet: {_label(n)[:25]}"
        for n in list(tainted_neighbors)[:3]
    ]
    return score, reasons


def score_wallet(
    wallet: str,
    df: pd.DataFrame,
    G: nx.DiGraph,
    tainted_set: set,
) -> dict:
    """Compute the composite risk score for a single wallet."""
    w = wallet.lower()

    # Normalise addresses for reliable filtering
    df_w = df.copy()
    df_w["from"] = df_w["from"].str.lower()
    df_w["to"]   = df_w["to"].str.lower()

    s_cycle,   r_cycle   = _score_cycle(w, G)
    s_transit, r_transit = _score_transit(w, df_w)
    s_struct,  r_struct  = _score_structuring(w, df_w)
    s_out, s_in, r_out, r_in = _score_fan(w, G)
    s_night,   r_night   = _score_night(w, df_w)
    s_taint,   r_taint   = _score_tainted(w, G, tainted_set)

    raw_score = (
        WEIGHTS["cycle"]       * s_cycle   +
        WEIGHTS["transit"]     * s_transit +
        WEIGHTS["structuring"] * s_struct  +
        WEIGHTS["fan_out"]     * s_out     +
        WEIGHTS["fan_in"]      * s_in      +
        WEIGHTS["night"]       * s_night   +
        WEIGHTS["tainted"]     * s_taint
    )

    score = round(min(100, raw_score))

    if score >= 60:
        verdict, color = "DANGEROUS",    "#FF3B3B"
    elif score >= 30:
        verdict, color = "SUSPICIOUS",   "#FF9500"
    else:
        verdict, color = "CLEAN",        "#30D158"

    all_reasons = r_cycle + r_transit + r_struct + r_out + r_in + r_night + r_taint

    return {
        "wallet":  wallet,
        "label":   KNOWN_WALLETS.get(wallet, wallet[:14] + "..."),
        "score":   score,
        "verdict": verdict,
        "color":   color,
        "reasons": all_reasons,
        "breakdown": {
            "Transaction cycle":  round(s_cycle   * WEIGHTS["cycle"]),
            "Fast transit":       round(s_transit  * WEIGHTS["transit"]),
            "Structuring":        round(s_struct   * WEIGHTS["structuring"]),
            "Fan-out":            round(s_out      * WEIGHTS["fan_out"]),
            "Fan-in":             round(s_in       * WEIGHTS["fan_in"]),
            "Night activity":     round(s_night    * WEIGHTS["night"]),
            "Tainted links":      round(s_taint    * WEIGHTS["tainted"]),
        },
    }


def run_scoring(
    wallets: list[str], df: pd.DataFrame, G: nx.DiGraph
) -> list[dict]:
    """Score all wallets iteratively; flagged wallets influence neighbours' scores."""
    results: list[dict] = []
    tainted: set = set()

    for wallet in wallets:
        res = score_wallet(wallet, df, G, tainted)
        results.append(res)
        if res["score"] >= 30:
            tainted.add(wallet.lower())

    return sorted(results, key=lambda x: x["score"], reverse=True)

ICONS = {"DANGEROUS": "<!>", "SUSPICIOUS": "/!\\", "CLEAN": "[OK]"}
_WEIGHT_LOOKUP = {
    "Transaction cycle": "cycle",
    "Fast transit":      "transit",
    "Structuring":       "structuring",
    "Fan-out":           "fan_out",
    "Fan-in":            "fan_in",
    "Night activity":    "night",
    "Tainted links":     "tainted",
}


def _risk_bar(score: int, width: int = 20) -> str:
    """Return a fixed-width ASCII progress bar."""
    filled = round(score / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def write_report(results: list[dict], df: pd.DataFrame, path: Path) -> None:
    """Write the full AML report to a plain-text file."""
    sep  = "=" * 72
    sep2 = "-" * 72
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    dangerous  = [r for r in results if r["verdict"] == "DANGEROUS"]
    suspicious = [r for r in results if r["verdict"] == "SUSPICIOUS"]
    clean      = [r for r in results if r["verdict"] == "CLEAN"]

    lines: list[str] = []

    # Header
    lines += [
        sep,
        "  ETH AML DETECTOR — WALLET RISK ASSESSMENT REPORT",
        f"  Generated : {now}",
        f"  Wallets   : {len(results)}",
        f"  Period    : {df['timeStamp'].min().date()} — {df['timeStamp'].max().date()}",
        f"  Tx total  : {len(df)}",
        sep,
        "",
        "  SUMMARY",
        sep2,
        f"  DANGEROUS  : {len(dangerous):>3}  wallet(s)",
        f"  SUSPICIOUS : {len(suspicious):>3}  wallet(s)",
        f"  CLEAN      : {len(clean):>3}  wallet(s)",
        "",
    ]

    # Per-wallet detail
    lines += [sep, "  DETAILED RESULTS (sorted by risk, highest first)", sep]

    for r in results:
        icon  = ICONS[r["verdict"]]
        bar   = _risk_bar(r["score"])
        lines += [
            "",
            f"  {icon} {r['label']}",
            f"      Address : {r['wallet']}",
            f"      Risk    : {bar} {r['score']:>3}/100  [{r['verdict']}]",
        ]

        if r["reasons"]:
            lines.append("      Findings:")
            for reason in r["reasons"]:
                # Wrap long lines neatly
                wrapped = textwrap.fill(reason, width=60,
                                        initial_indent="        * ",
                                        subsequent_indent="          ")
                lines.append(wrapped)
        else:
            lines.append("      Findings : No anomalies detected.")

        lines.append("      Score breakdown:")
        for feature, pts in r["breakdown"].items():
            max_pts = WEIGHTS.get(_WEIGHT_LOOKUP.get(feature, ""), 0)
            mini    = "#" * pts + "-" * max(0, max_pts - pts)
            lines.append(f"        {feature:<22}  +{pts:>2} / {max_pts:<2}  [{mini}]")

        lines.append(sep2)

    # Flagged wallets summary
    lines += [
        "",
        sep,
        "  FLAGGED WALLETS — REQUIRE INVESTIGATION",
        sep,
    ]

    if dangerous or suspicious:
        for r in dangerous + suspicious:
            lines += [
                f"  {ICONS[r['verdict']]} {r['wallet']}",
                f"      {r['label']}  |  Risk: {r['score']}/100  [{r['verdict']}]",
            ]
            for reason in r["reasons"][:2]:
                lines.append(f"      -> {reason}")
            lines.append("")
    else:
        lines += ["  No flagged wallets found.", ""]

    lines += [sep, "  END OF REPORT", sep, ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Report saved to: {path.resolve()}")

def _label(addr: str) -> str:
    """Return a short human-readable label for a wallet address."""
    for k, v in KNOWN_WALLETS.items():
        if k.lower() == addr.lower():
            return v.split("(")[0].strip()
    return addr[:6] + "…" + addr[-4:]


def visualize(
    G: nx.DiGraph, results: list[dict], tracked_wallets: set
) -> None:
    """Render the transaction graph and risk-rating bar chart; save as PNG."""
    if G.number_of_nodes() == 0:
        print("[!] Graph is empty — nothing to visualise.")
        return

    score_map = {r["wallet"].lower(): r for r in results}

    # Node appearance based on risk verdict
    node_colors, node_sizes = [], []
    for n in G.nodes():
        res = score_map.get(n)
        if res:
            node_colors.append(res["color"])
            node_sizes.append(600 + res["score"] * 8)
        elif n in tracked_wallets:
            node_colors.append("#0A84FF")
            node_sizes.append(600)
        else:
            node_colors.append("#3A3A5C")
            node_sizes.append(300)

    # Edge width proportional to ETH volume
    values      = [d["value"] for _, _, d in G.edges(data=True)]
    max_vol     = max(values) if values else 1
    edge_widths = [0.4 + 4 * (d["value"] / max_vol) for _, _, d in G.edges(data=True)]

    # Choose layout algorithm by graph size
    n = G.number_of_nodes()
    if n <= 20:
        pos = nx.spring_layout(G, seed=42, k=2.5)
    elif n <= 70:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.0, iterations=40)

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 10),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.patch.set_facecolor("#0D1117")

    # Left panel: transaction graph
    ax = axes[0]
    ax.set_facecolor("#0D1117")

    nx.draw_networkx_edges(
        G, pos, ax=ax, width=edge_widths, edge_color="#58A6FF",
        alpha=0.5, arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors,
        node_size=node_sizes, edgecolors="#FFFFFF", linewidths=0.6,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: _label(n) for n in G.nodes()},
        font_size=6, font_color="#E6EDF3",
    )

    legend_items = [
        mpatches.Patch(color="#FF3B3B", label="DANGEROUS (60–100)"),
        mpatches.Patch(color="#FF9500", label="SUSPICIOUS (30–59)"),
        mpatches.Patch(color="#30D158", label="CLEAN (0–29)"),
        mpatches.Patch(color="#0A84FF", label="Tracked wallet"),
        mpatches.Patch(color="#3A3A5C", label="External address"),
    ]
    ax.legend(
        handles=legend_items, loc="upper left",
        facecolor="#161B22", edgecolor="#30363D",
        labelcolor="#E6EDF3", fontsize=8,
    )
    ax.set_title("Transaction Graph", color="#E6EDF3", fontsize=12, pad=10)

    # Right panel: risk rating bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#0D1117")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.5, len(results) - 0.5)
    ax2.axis("off")
    ax2.set_title("Wallet Risk Ranking", color="#E6EDF3", fontsize=12, pad=10)

    for i, r in enumerate(reversed(results)):
        y, score, color, label = i, r["score"], r["color"], r["label"][:28]

        ax2.barh(y, 100,   color="#161B22", height=0.7, left=0)          # background
        ax2.barh(y, score, color=color,     height=0.7, left=0, alpha=0.85)  # risk bar
        ax2.text(-1, y, label, va="center", ha="right",
                 color="#E6EDF3", fontsize=7.5)
        ax2.text(score + 1, y, str(score), va="center",
                 color=color, fontsize=8, fontweight="bold")
        ax2.text(96, y, ICONS[r["verdict"]], va="center",
                 ha="center", fontsize=8)

    # Threshold lines
    ax2.axvline(30, color="#FF9500", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axvline(60, color="#FF3B3B", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.text(30, -0.5, "30", color="#FF9500", fontsize=7, ha="center")
    ax2.text(60, -0.5, "60", color="#FF3B3B", fontsize=7, ha="center")

    plt.tight_layout()
    out_png = "eth_aml_report.png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    print(f"[OK] Graph saved to: {out_png}")
    plt.show()

if __name__ == "__main__":
    print("=" * 68)
    print("  ETH AML Detector — Anti-Money Laundering Analysis")
    print(f"  Wallets to analyse: {len(WALLETS)}")
    print("=" * 68)

    # Step 1 — fetch transaction data
    print("\n[1/4] Fetching transactions...")
    df = fetch_all_wallets(WALLETS)
    if df.empty:
        print("[!] No data retrieved. Check your API key.")
        raise SystemExit(1)
    print(
        f"[OK] Transactions: {len(df)}  |  "
        f"Period: {df['timeStamp'].min().date()} — {df['timeStamp'].max().date()}"
    )

    # Step 2 — build transaction graph
    print("\n[2/4] Building transaction graph...")
    G = build_graph(df)
    print(f"[OK] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Step 3 — compute risk scores and write report
    print("\n[3/4] Computing risk scores...")
    results = run_scoring(WALLETS, df, G)
    write_report(results, df, REPORT_PATH)

    # Step 4 — visualise
    print("\n[4/4] Building visualisation...")
    tracked = {w.lower() for w in WALLETS}
    visualize(G, results, tracked)