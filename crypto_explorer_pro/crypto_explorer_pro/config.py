"""Stałe konfiguracyjne — Crypto Explorer Pro."""

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

COINS = {
    "Bitcoin (BTC)":    "bitcoin",
    "Ethereum (ETH)":   "ethereum",
    "BNB (BNB)":        "binancecoin",
    "Solana (SOL)":     "solana",
    "Cardano (ADA)":    "cardano",
    "Avalanche (AVAX)": "avalanche-2",
    "Polygon (MATIC)":  "matic-network",
    "Chainlink (LINK)": "chainlink",
    "Uniswap (UNI)":    "uniswap",
    "Aave (AAVE)":      "aave",
    "Maker (MKR)":      "maker",
    "Dogecoin (DOGE)":  "dogecoin",
}

# Etykieta → liczba dni (None = zakres niestandardowy)
RANGES = {
    "24 godziny": 1,
    "7 dni":      7,
    "30 dni":     30,
    "90 dni":     90,
    "1 rok":      365,
}

C_BG     = "#0d0d0d"
C_PANEL  = "#141414"
C_PANEL2 = "#1a1a1a"
C_GRID   = "#222222"
C_BORDER = "#2a2a2a"
C_TEXT   = "#cccccc"
C_TEXT2  = "#888888"
C_TEXT3  = "#555555"

C_UP     = "#00c896"   # zielony — KUP
C_DOWN   = "#ff4d4d"   # czerwony — SPRZEDAJ
C_MA7    = "#f5a623"   # pomarańczowy
C_MA25   = "#a78bfa"   # fioletowy
C_BB     = "#4a90d9"   # niebieski — Bollinger
C_VOL    = "#3a4a6a"   # niebieskoszary — wolumen

# Tryby KUP / SPRZEDAJ
MODES = {
    "buy":  {"label": "▲  KUP",      "color": C_UP,   "dim": "#003d2e"},
    "sell": {"label": "▼  SPRZEDAJ", "color": C_DOWN, "dim": "#3d0a0a"},
}
