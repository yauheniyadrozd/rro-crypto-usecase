"""
ETH AML — Side-by-Side Comparison
===================================
Runs the rule-based detector AND the LSTM detector, then writes a
merged JSON file consumed by the HTML comparison dashboard.

Usage
-----
  python eth_aml_compare.py

Prerequisites
-------------
  • Both eth_aml_detector.py and eth_aml_lstm.py are in the same folder.
  • ETHERSCAN_API_KEY is set in BOTH scripts.
  • pip install numpy pandas scikit-learn torch networkx requests matplotlib

What this script does
---------------------
  1. Imports the core logic from both detectors (no duplicate API calls —
     transactions are fetched once and shared).
  2. Runs rule-based scoring.
  3. Builds feature sequences and trains the LSTM on pseudo-labels.
  4. Produces eth_aml_comparison.json  ← consumed by dashboard.html
  5. Prints a compact terminal diff table.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Bootstrap: import the two detector modules ─────────────────────────

def _load(path: str):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

BASE_DIR = Path(__file__).parent
det_path  = BASE_DIR / "eth_aml_detector.py"
lstm_path = BASE_DIR / "eth_aml_lstm.py"

if not det_path.exists():
    sys.exit(f"[!] Cannot find {det_path}")
if not lstm_path.exists():
    sys.exit(f"[!] Cannot find {lstm_path}")

print("[…] Loading detector modules…")
det  = _load(str(det_path))
lstm = _load(str(lstm_path))

# ── Shared configuration ───────────────────────────────────────────────

WALLETS      = det.WALLETS         # canonical wallet list
KNOWN_WALLETS = det.KNOWN_WALLETS
OUTPUT_JSON  = BASE_DIR / "eth_aml_comparison.json"


# ── Step 1: Fetch data ONCE ────────────────────────────────────────────

print("\n[1/5] Fetching transactions & token transfers (shared)…")
fetch_result = det.fetch_all_wallets(WALLETS)
# Support both old (DataFrame) and new (DataFrame, token_map) return shapes
if isinstance(fetch_result, tuple):
    df, token_map = fetch_result
else:
    df = fetch_result
    # Fetch token transfers via lstm module (has get_token_transfers)
    token_map = {}
    for addr in WALLETS:
        token_map[addr.lower()] = lstm.get_token_transfers(addr)
        import time; time.sleep(0.25)

if df.empty:
    sys.exit("[!] No transactions retrieved. Check your API key.")
print(f"[OK] {len(df):,} transactions loaded.")


# ── Step 2: Rule-based scoring ─────────────────────────────────────────

print("\n[2/5] Running rule-based detector…")
G           = det.build_graph(df)
rule_results = det.run_scoring(WALLETS, df, G)

# Normalise address case for lookup
rule_map = {r["wallet"].lower(): r for r in rule_results}


# ── Step 3: LSTM scoring ───────────────────────────────────────────────

print("\n[3/5] Building LSTM sequences & pseudo-labels…")
wallets_lower = [w.lower() for w in WALLETS]
X, y = lstm.build_dataset(wallets_lower, df)

print("\n[4/5] Training BiLSTM…")
model = lstm.train_model(X, y)

print("\n[4b/5] Scoring with LSTM…")
lstm_results = lstm.score_all_wallets(wallets_lower, df, model, X, token_map)
lstm_map     = {r["wallet"].lower(): r for r in lstm_results}


# ── Expected verdicts (ground truth for dashboard validation) ──────────
# Keys are lowercase addresses. Values: (expected_verdict, explanation)

EXPECTED_VERDICTS: dict[str, tuple[str, str]] = {
    "0xd8da6bf26964af9d7eed9e03e53415d37aa96045": (
        "CLEAN",
        "Well-known public address. Donations, grants, public transfers only. No illicit links."),
    "0xab5801a7d398351b8be11c439e05c5b3259aec9b": (
        "CLEAN",
        "Secondary public address, low activity, no suspicious patterns documented."),
    "0x220866b1a2219f40e72f5c628b65d54268ca3a9d": (
        "CLEAN",
        "Tertiary address. Night activity likely reflects timezone of use, not automation."),
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": (
        "SUSPICIOUS",
        "Licensed exchange — high volume is expected and legal, but pattern complexity warrants monitoring. Not DANGEROUS despite volume."),
    "0xd551234ae421e3bcba99a0da6d736074f22192ff": (
        "SUSPICIOUS",
        "CEX hot wallet — fast transit is normal exchange mechanic, not structuring. Should not be DANGEROUS."),
    "0x564286362092d8e7936f0549571a803b203aaced": (
        "SUSPICIOUS",
        "Night activity reflects 24/7 exchange operations across timezones — not inherently suspicious for a CEX."),
    "0xf977814e90da44bfa03b6295a0616a897441acec": (
        "SUSPICIOUS",
        "Round-number clustering is a Binance internal accounting pattern, not structuring."),
    "0x28c6c06298d514db089934071355e5743bf21d60": (
        "SUSPICIOUS",
        "Normal exchange hot wallet behaviour. Fan-in from retail users, redistribution to cold storage."),
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": (
        "SUSPICIOUS",
        "Licensed US exchange (publicly traded). Volume expected. Should not exceed SUSPICIOUS."),
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": (
        "SUSPICIOUS",
        "Secondary hot wallet. Fast transit at typical CEX levels, not mixer-level."),
    "0x77696bb39917c91a0c3908d577d5e322095425ca": (
        "SUSPICIOUS",
        "Tertiary Coinbase hot wallet — low anomaly score expected."),
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": (
        "SUSPICIOUS",
        "Licensed EU exchange. 00:00 UTC bursts are end-of-day settlement — standard treasury operation."),
    "0xae2d4617c862309a3d75a0ffb358c7a5009c673f": (
        "SUSPICIOUS",
        "Low-activity Kraken wallet. Both models close to SUSPICIOUS/CLEAN boundary — acceptable."),
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": (
        "SUSPICIOUS",
        "OKX operates globally including Asia — high night activity (UTC) is expected for an Asian exchange."),
    "0x236f233dbf88d2a741e8a4e46b0cf2b3afa11e68": (
        "CLEAN",
        "Cold storage address. Infrequent large deposits only. Both models should score very low."),
    "0x00000000219ab540356cbb839cbe05303d7705fa": (
        "CLEAN",
        "Official ETH staking contract. Every deposit is exactly 32 ETH — mechanical, not suspicious. Rules may over-flag fan-in."),
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": (
        "CLEAN",
        "Core DeFi infrastructure. Wrap/unwrap is neutral activity. Both models should stay below SUSPICIOUS."),
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": (
        "CLEAN",
        "Immutable AMM smart contract, open source, no custody. Routing is not a layering risk."),
    "0xe592427a0aece92de3edee1f18e0157c05861564": (
        "CLEAN",
        "Protocol-level contract. Should be CLEAN across both models."),
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff": (
        "CLEAN",
        "DEX aggregator. Routing through multiple sources superficially resembles layering — models may flag, but it is a false positive."),
    "0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3": (
        "DANGEROUS",
        "OFAC sanctioned since Aug 2022. Fixed-denomination mixer. Both models must score 80+. Any lower score is a model failure."),
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf": (
        "DANGEROUS",
        "OFAC sanctioned. Identical denomination mixer. Score below 80 = model failure."),
    "0xbb93e510bbcd0b7beb5a853875f9ec60275cf498": (
        "DANGEROUS",
        "OFAC sanctioned. Highest structuring signal expected. Score below 75 = model failure."),
    "0x8589427373d6d84e98730d7795d8f6f8731fda16": (
        "DANGEROUS",
        "OFAC sanctioned. Micro-denomination pool. Score below 70 = model failure."),
    "0x1919db36ca2fa2e15f9000fd9cdc2edcf863e685": (
        "SUSPICIOUS",
        "Possible wash-trading pattern. LSTM should elevate score vs rules. CLEAN = under-detection; DANGEROUS = over-detection."),
}

# ── Step 5: Merge & compare ────────────────────────────────────────────

print("\n[5/5] Merging results…")
comparison: list[dict] = []

for w in wallets_lower:
    rr = rule_map.get(w, {})
    lr = lstm_map.get(w, {})
    label = KNOWN_WALLETS.get(w, KNOWN_WALLETS.get(
        next((k for k in KNOWN_WALLETS if k.lower() == w), ""), w[:14] + "..."
    ))

    rule_score   = rr.get("score", 0)
    lstm_score   = lr.get("score", 0)
    rule_verdict = rr.get("verdict", "N/A")
    lstm_verdict = lr.get("verdict", "N/A")
    delta        = lstm_score - rule_score
    agreement    = abs(delta) <= 15

    exp_tuple     = EXPECTED_VERDICTS.get(w)
    expected      = exp_tuple[0] if exp_tuple else None
    expected_note = exp_tuple[1] if exp_tuple else None
    rule_ok       = (rule_verdict == expected) if expected else None
    lstm_ok       = (lstm_verdict == expected) if expected else None

    comparison.append({
        "wallet":       w,
        "label":        rr.get("label", label),
        "expected":     expected,
        "expectedNote": expected_note,
        "ruleOk":       rule_ok,
        "lstmOk":       lstm_ok,
        "rule": {
            "score":     rule_score,
            "verdict":   rule_verdict,
            "color":     rr.get("color", "#888"),
            "reasons":   rr.get("reasons", []),
            "breakdown": rr.get("breakdown", {}),
        },
        "lstm": {
            "score":     lstm_score,
            "verdict":   lstm_verdict,
            "color":     lr.get("color", "#888"),
            "reasons":   lr.get("reasons", []),
            "breakdown": lr.get("breakdown", {}),
            "prob":      lr.get("prob", 0.0),
        },
        "delta":     delta,
        "agreement": agreement,
        "tokens":    token_map.get(w, []),
    })

# Sort by average risk descending
comparison.sort(key=lambda x: (x["rule"]["score"] + x["lstm"]["score"]) / 2,
                reverse=True)

# Method-level statistics
rule_scores = [c["rule"]["score"] for c in comparison]
lstm_scores = [c["lstm"]["score"] for c in comparison]

meta = {
    "generated":       datetime.now(timezone.utc).isoformat(),
    "n_wallets":       len(comparison),
    "n_transactions":  len(df),
    "date_range": {
        "start": str(df["timeStamp"].min().date()),
        "end":   str(df["timeStamp"].max().date()),
    },
    "rule_stats": {
        "mean":      round(float(np.mean(rule_scores)), 1),
        "max":       int(np.max(rule_scores)),
        "dangerous": sum(1 for s in rule_scores if s >= 60),
        "suspicious":sum(1 for s in rule_scores if 30 <= s < 60),
        "clean":     sum(1 for s in rule_scores if s < 30),
    },
    "lstm_stats": {
        "mean":      round(float(np.mean(lstm_scores)), 1),
        "max":       int(np.max(lstm_scores)),
        "dangerous": sum(1 for s in lstm_scores if s >= 60),
        "suspicious":sum(1 for s in lstm_scores if 30 <= s < 60),
        "clean":     sum(1 for s in lstm_scores if s < 30),
    },
    "agreement_rate": round(
        sum(1 for c in comparison if c["agreement"]) / len(comparison) * 100, 1
    ),
}

output = {"meta": meta, "wallets": comparison}
OUTPUT_JSON.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

# ── Terminal diff table ────────────────────────────────────────────────

print("\n" + "=" * 80)
print(f"  {'WALLET':<32}  {'RULE':>5}  {'LSTM':>5}  {'Δ':>5}  {'AGREE':>6}")
print("=" * 80)
for c in comparison:
    agree = "✓" if c["agreement"] else "✗"
    delta_str = f"+{c['delta']}" if c["delta"] > 0 else str(c["delta"])
    print(f"  {c['label'][:32]:<32}  {c['rule']['score']:>5}  "
          f"{c['lstm']['score']:>5}  {delta_str:>5}  {agree:>6}")

print("=" * 80)
print(f"  Rule-based mean: {meta['rule_stats']['mean']:>5}   "
      f"LSTM mean: {meta['lstm_stats']['mean']:>5}   "
      f"Agreement: {meta['agreement_rate']}%")
print(f"\n[OK] Comparison saved -> {OUTPUT_JSON.resolve()}")
print("[→]  Open eth_aml_dashboard.html in your browser to view the dashboard.")