"""
ETH AML Detector — LSTM Edition
================================
Trains a bidirectional LSTM on transaction sequences per wallet,
produces risk scores and explanations, and saves results to JSON
so the comparison script can load both approaches side-by-side.

Architecture
------------
  For each wallet we build a time-ordered sequence of feature vectors,
  one vector per transaction:
    [value_eth, hour, is_outgoing, counterparty_entropy, cumulative_out,
     rolling_velocity, is_night, log_value]

  A Bidirectional LSTM reads the full sequence and outputs a single
  risk probability in [0, 1].  A simple MLP head converts the hidden
  state to a score.

  Because real labels are unavailable for arbitrary wallets we use a
  SELF-SUPERVISED strategy:
    1. Extract heuristic pseudo-labels from the rule-based scorer.
    2. Train the LSTM on those pseudo-labels with cross-entropy loss.
    3. Re-score every wallet with the trained model.
  This lets the LSTM generalise beyond the fixed rules while remaining
  interpretable through attention-weight visualisation.

Dependencies
------------
  pip install numpy pandas scikit-learn torch requests networkx
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ETHERSCAN_API_KEY = os.environ["ETHERSCAN_API_KEY"]

KNOWN_WALLETS: dict[str, str] = {
    # Public figures
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

WALLETS: list[str] = list(KNOWN_WALLETS.keys())

MIN_VALUE_ETH          = 0.05
DELAY_BETWEEN_REQUESTS = 0.25
NIGHT_HOURS            = (2, 6)
SEQUENCE_LEN           = 64      # pad / truncate to this many transactions
FEATURE_DIM            = 9       # features per transaction step
HIDDEN_DIM             = 64
NUM_LAYERS             = 2
DROPOUT                = 0.3
LR                     = 1e-3
EPOCHS                 = 30
BATCH_SIZE             = 4       # small — few wallets

RESULTS_JSON = Path("eth_aml_lstm_results.json")

def get_wallet_transactions(address: str) -> pd.DataFrame:
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": 1, "module": "account", "action": "txlist",
        "address": address, "startblock": 0, "endblock": 99_999_999,
        "sort": "asc", "apikey": ETHERSCAN_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    [!] {exc}")
        return pd.DataFrame()

    if data["status"] != "1":
        return pd.DataFrame()

    df = pd.DataFrame(data["result"])
    if df.empty:
        return df

    df["value_eth"] = df["value"].astype(float) / 1e18
    df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s", utc=True)
    df["from"] = df["from"].str.lower()
    df["to"]   = df["to"].str.lower()
    df["wallet"] = address.lower()
    return df[["timeStamp", "from", "to", "value_eth", "hash", "wallet"]]


def get_token_transfers(address: str) -> list[dict]:
    """Fetch ERC-20 token transfers for a wallet and return a summary list."""
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": 1, "module": "account", "action": "tokentx",
        "address": address, "startblock": 0, "endblock": 99_999_999,
        "sort": "desc", "apikey": ETHERSCAN_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    if data["status"] != "1" or not data.get("result"):
        return []

    rows = data["result"]
    # Aggregate by token symbol
    token_stats: dict[str, dict] = {}
    addr_lower = address.lower()
    for row in rows:
        symbol   = row.get("tokenSymbol", "?") or "?"
        name     = row.get("tokenName",   "?") or "?"
        decimals = int(row.get("tokenDecimal", 18) or 18)
        try:
            amount = int(row.get("value", 0)) / (10 ** decimals)
        except Exception:
            amount = 0.0
        direction = "in" if row.get("to", "").lower() == addr_lower else "out"

        if symbol not in token_stats:
            token_stats[symbol] = {
                "symbol": symbol,
                "name": name,
                "in_count": 0, "out_count": 0,
                "in_amount": 0.0, "out_amount": 0.0,
            }
        if direction == "in":
            token_stats[symbol]["in_count"]  += 1
            token_stats[symbol]["in_amount"] += amount
        else:
            token_stats[symbol]["out_count"]  += 1
            token_stats[symbol]["out_amount"] += amount

    # Sort by total tx count descending, return top 15
    result = sorted(token_stats.values(),
                    key=lambda x: x["in_count"] + x["out_count"], reverse=True)
    return result[:15]


def fetch_all_wallets(wallets: list[str]) -> tuple[pd.DataFrame, dict[str, list]]:
    """Returns (tx_dataframe, token_map) where token_map is addr -> token list."""
    all_dfs, token_map = [], {}
    for i, addr in enumerate(wallets, 1):
        label = KNOWN_WALLETS.get(addr, addr[:14] + "...")
        print(f"  [{i}/{len(wallets)}] {label}")
        df = get_wallet_transactions(addr)
        if not df.empty:
            all_dfs.append(df)
            print(f"         -> {len(df):,} txs")
        time.sleep(DELAY_BETWEEN_REQUESTS)
        tokens = get_token_transfers(addr)
        token_map[addr.lower()] = tokens
        if tokens:
            syms = ", ".join(t["symbol"] for t in tokens[:5])
            print(f"         -> tokens: {syms}{' ...' if len(tokens) > 5 else ''}")
        time.sleep(DELAY_BETWEEN_REQUESTS)
    if not all_dfs:
        return pd.DataFrame(), token_map
    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("hash")
    return combined, token_map

def _counterparty_entropy(wallet: str, df: pd.DataFrame) -> pd.Series:
    """Rolling Shannon entropy of counterparty addresses (diversity measure)."""
    wallet_txs = df[(df["from"] == wallet) | (df["to"] == wallet)].copy()
    wallet_txs["cp"] = np.where(wallet_txs["from"] == wallet,
                                wallet_txs["to"], wallet_txs["from"])
    # rolling unique counterparties over last 10 txs
    def rolling_entropy(series: pd.Series, window: int = 10) -> pd.Series:
        result = []
        for i in range(len(series)):
            chunk = series.iloc[max(0, i - window + 1): i + 1]
            counts = chunk.value_counts(normalize=True)
            h = -float((counts * np.log2(counts + 1e-9)).sum())
            result.append(h)
        return pd.Series(result, index=series.index)
    return rolling_entropy(wallet_txs["cp"])


def build_sequence(wallet: str, df: pd.DataFrame, seq_len: int) -> np.ndarray:
    """
    Build a (seq_len, FEATURE_DIM) matrix for one wallet.

    Features per step:
      0  log1p(value_eth)
      1  hour / 23                          (normalised hour of day)
      2  is_outgoing                         (1 = sent, 0 = received)
      3  rolling_counterparty_entropy / 4    (normalised)
      4  cumulative_eth_out / (cum_out+1)    (proportion out so far)
      5  inter-tx gap in hours / 24          (time since previous tx)
      6  is_night                            (1 = 02-06 UTC)
      7  value_eth / (max_value + 1e-9)      (relative value)
      8  round_number flag                   (value divisible by 0.1 ETH)
    """
    w = wallet.lower()
    wallet_txs = df[(df["from"] == w) | (df["to"] == w)].copy()
    wallet_txs = wallet_txs.sort_values("timeStamp").reset_index(drop=True)

    if wallet_txs.empty:
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)

    max_val = wallet_txs["value_eth"].max() + 1e-9
    cum_out = 0.0
    prev_ts = None
    entropy_series = _counterparty_entropy(w, wallet_txs)

    rows: list[list[float]] = []
    for i, row in wallet_txs.iterrows():
        is_out = float(row["from"] == w)
        v      = row["value_eth"]
        if is_out:
            cum_out += v
        hour   = row["timeStamp"].hour
        is_night = float(NIGHT_HOURS[0] <= hour <= NIGHT_HOURS[1])
        gap    = 0.0 if prev_ts is None else (
            (row["timeStamp"] - prev_ts).total_seconds() / 3600
        )
        prev_ts = row["timeStamp"]
        ent    = float(entropy_series.iloc[i]) / 4.0 if i < len(entropy_series) else 0.0
        cum_r  = cum_out / (cum_out + 1.0)
        round_flag = float(abs(v % 0.1) < 0.001)

        rows.append([
            math.log1p(v),
            hour / 23.0,
            is_out,
            min(ent, 1.0),
            cum_r,
            min(gap / 24.0, 1.0),
            is_night,
            v / max_val,
            round_flag,
        ])

    arr = np.array(rows, dtype=np.float32)

    # Truncate or zero-pad to seq_len
    if len(arr) >= seq_len:
        return arr[-seq_len:]           # keep most recent transactions
    pad = np.zeros((seq_len - len(arr), FEATURE_DIM), dtype=np.float32)
    return np.vstack([pad, arr])

def heuristic_score(wallet: str, df: pd.DataFrame) -> float:
    """
    Lightweight rule-based score in [0, 1] used as pseudo-label.
    (Simplified subset of the full rule-based scorer.)
    """
    w = wallet.lower()
    wallet_txs = df[(df["from"] == w) | (df["to"] == w)]
    if wallet_txs.empty:
        return 0.0

    total = len(wallet_txs)
    score = 0.0

    # Night activity
    night = wallet_txs[wallet_txs["timeStamp"].dt.hour.between(*NIGHT_HOURS)]
    night_ratio = len(night) / total
    if night_ratio > 0.4:
        score += 0.10 * min(1.0, (night_ratio - 0.4) / 0.4)

    # Fan-out (unique recipients)
    sent = wallet_txs[wallet_txs["from"] == w]
    n_recipients = sent["to"].nunique() if not sent.empty else 0
    score += 0.15 * min(1.0, max(0, n_recipients - 10) / 10)

    # Fan-in (unique senders)
    recv = wallet_txs[wallet_txs["to"] == w]
    n_senders = recv["from"].nunique() if not recv.empty else 0
    score += 0.15 * min(1.0, max(0, n_senders - 10) / 10)

    # Structuring (repeat transfers to same address)
    if not sent.empty:
        repeat = sent.groupby("to").size()
        n_structured = (repeat >= 5).sum()
        score += 0.20 * min(1.0, n_structured / 5)

    # Fast transit (in-and-out within 1 hour)
    incoming = wallet_txs[wallet_txs["to"] == w].sort_values("timeStamp")
    outgoing = wallet_txs[wallet_txs["from"] == w].sort_values("timeStamp")
    transit = 0
    for _, inc in incoming.iterrows():
        window = inc["timeStamp"] + pd.Timedelta(hours=1)
        if not outgoing[
            (outgoing["timeStamp"] >= inc["timeStamp"]) &
            (outgoing["timeStamp"] <= window)
        ].empty:
            transit += 1
    score += 0.25 * min(1.0, transit / max(len(incoming), 1))

    # Round-number transfers (structuring indicator)
    round_txs = sent[sent["value_eth"].apply(lambda v: abs(v % 0.1) < 0.001)]
    round_ratio = len(round_txs) / max(len(sent), 1)
    score += 0.15 * min(1.0, round_ratio)

    return float(np.clip(score, 0.0, 1.0))


class AMLBiLSTM(nn.Module):
    """
    Bidirectional LSTM with self-attention pooling and MLP head.

    Attention pooling lets us extract per-step importance weights,
    which we later use to highlight the most suspicious transactions.
    """

    def __init__(
        self,
        input_dim: int  = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float  = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)   # attention scoring
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)                        # (batch, seq, 2*H)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (batch, seq, 1)
        ctx    = (attn_w * out).sum(dim=1)           # (batch, 2*H)
        prob   = self.head(ctx).squeeze(-1)          # (batch,)
        if return_attention:
            return prob, attn_w.squeeze(-1)          # attn: (batch, seq)
        return prob, None


def build_dataset(
    wallets: list[str], df: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X, y) tensors from transaction sequences and pseudo-labels."""
    X_list, y_list = [], []
    for w in wallets:
        seq   = build_sequence(w, df, SEQUENCE_LEN)     # (seq_len, features)
        label = heuristic_score(w, df)
        X_list.append(seq)
        y_list.append(label)
        print(f"    {KNOWN_WALLETS.get(w, w[:14]+'...'):<32}  pseudo-label={label:.3f}")
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)  # (N, seq, feat)
    y = torch.tensor(y_list, dtype=torch.float32)             # (N,)
    return X, y


def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = EPOCHS,
) -> AMLBiLSTM:
    model     = AMLBiLSTM()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.BCELoss()
    dataset   = TensorDataset(X, y)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}/{epochs}  loss={epoch_loss/len(loader):.4f}")

    model.eval()
    return model


def _attention_to_reasons(
    wallet: str,
    df: pd.DataFrame,
    attn_weights: np.ndarray,
    top_k: int = 3,
) -> list[str]:
    """Map high-attention time steps back to transaction descriptions."""
    w = wallet.lower()
    wallet_txs = (
        df[(df["from"] == w) | (df["to"] == w)]
        .sort_values("timeStamp")
        .reset_index(drop=True)
    )
    if wallet_txs.empty:
        return []

    seq_len = len(attn_weights)
    n_real  = len(wallet_txs)

    # Align attention weights with actual transactions (padding at start)
    offset = seq_len - n_real  # number of zero-padding steps
    aligned = []
    for i, row in wallet_txs.iterrows():
        attn_idx = offset + i
        if 0 <= attn_idx < seq_len:
            aligned.append((attn_weights[attn_idx], row))

    aligned.sort(key=lambda x: x[0], reverse=True)
    reasons = []
    for attn_val, row in aligned[:top_k]:
        direction = "Sent" if row["from"] == w else "Received"
        cp        = row["to"] if row["from"] == w else row["from"]
        cp_label  = cp[:8] + "…" + cp[-4:]
        ts        = row["timeStamp"].strftime("%Y-%m-%d %H:%M UTC")
        reasons.append(
            f"[attn={attn_val:.3f}] {direction} {row['value_eth']:.4f} ETH "
            f"{'to' if direction=='Sent' else 'from'} {cp_label} at {ts}"
        )
    return reasons


def score_all_wallets(
    wallets: list[str],
    df: pd.DataFrame,
    model: AMLBiLSTM,
    X: torch.Tensor,
    token_map: dict | None = None,
) -> list[dict]:
    model.eval()
    results = []
    with torch.no_grad():
        probs, attn_weights = model(X, return_attention=True)

    for i, wallet in enumerate(wallets):
        prob  = float(probs[i])
        score = round(prob * 100)
        attn  = attn_weights[i].cpu().numpy()

        if score >= 60:
            verdict, color = "DANGEROUS",  "#FF3B3B"
        elif score >= 30:
            verdict, color = "SUSPICIOUS", "#FF9500"
        else:
            verdict, color = "CLEAN",      "#30D158"

        reasons = _attention_to_reasons(wallet, df, attn)

        seq = X[i].numpy()
        feat_importance = (attn[:, None] * np.abs(seq)).mean(axis=0)
        feat_labels = [
            "Log value", "Hour of day", "Outgoing", "CP entropy",
            "Cumul. out", "TX gap", "Night flag", "Relative value", "Round num",
        ]
        breakdown = {
            feat_labels[j]: round(float(feat_importance[j]) * 100, 2)
            for j in range(FEATURE_DIM)
        }

        results.append({
            "wallet":    wallet,
            "label":     KNOWN_WALLETS.get(wallet, wallet[:14] + "..."),
            "score":     score,
            "prob":      round(prob, 4),
            "verdict":   verdict,
            "color":     color,
            "reasons":   reasons,
            "breakdown": breakdown,
            "tokens":    (token_map or {}).get(wallet, []),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    print("=" * 68)
    print("  ETH AML Detector — LSTM Edition")
    print(f"  Wallets: {len(WALLETS)}")
    print("=" * 68)

    print("\n[1/4] Fetching transactions & token transfers...")
    df, token_map = fetch_all_wallets(WALLETS)
    if df.empty:
        print("[!] No data. Check your API key.")
        raise SystemExit(1)
    print(f"[OK] {len(df):,} transactions, "
          f"{df['timeStamp'].min().date()} — {df['timeStamp'].max().date()}")

    print("\n[2/4] Building sequences & pseudo-labels...")
    wallets_lower = [w.lower() for w in WALLETS]
    X, y = build_dataset(wallets_lower, df)
    print(f"[OK] X shape: {tuple(X.shape)}")

    print("\n[3/4] Training BiLSTM...")
    model = train_model(X, y)
    print("[OK] Training complete")

    print("\n[4/4] Scoring wallets...")
    results = score_all_wallets(wallets_lower, df, model, X, token_map)

    out = {
        "method":    "LSTM",
        "generated": datetime.now(timezone.utc).isoformat(),
        "results":   results,
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] Results saved -> {RESULTS_JSON.resolve()}")

    for r in results:
        bar = "#" * (r["score"] // 5) + "-" * (20 - r["score"] // 5)
        syms = ", ".join(t["symbol"] for t in r["tokens"][:4]) if r["tokens"] else "—"
        print(f"  [{bar}] {r['score']:>3}/100  {r['verdict']:<10}  {r['label']}  [{syms}]")