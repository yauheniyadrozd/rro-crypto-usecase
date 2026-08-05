# Cryptotest

Ethereum AML wallet risk analysis with rule-based heuristics and bidirectional LSTM, plus a crypto market explorer with robust optimization.

## Project Structure

```
cryptotest/
├── .env.example              # Environment variables template
├── .gitignore
├── requirements.txt           # Global dependencies
├── README.md
├── dashboard.html             # Interactive AML comparison dashboard
│
├── eth_aml/                   # Ethereum AML analysis
│   ├── eth_aml_detector.py    # Rule-based wallet risk scoring (7 heuristics)
│   ├── eth_aml_lstm.py        # Bidirectional LSTM risk scoring model
│   └── eth_aml_compare.py     # Side-by-side comparison + JSON export
│
├── crypto_explorer_pro/       # Crypto market analysis desktop app
│   ├── main.py                # Application entry point
│   ├── requirements.txt       # App-specific dependencies
│   └── crypto_explorer_pro/   # Application package
│       ├── __init__.py
│       ├── api.py             # CoinGecko API client
│       ├── app.py             # Application entry point (Tkinter)
│       ├── chart.py           # Main chart window with technical indicators
│       ├── config.py          # Color palette, coin list, constants
│       └── indicators.py      # Technical indicators (MA, BB, RSI)
│
├── output/                    # Analysis results (JSON)
│   ├── eth_aml_comparison.json
│   └── eth_aml_report.txt
│
├── reports/                   # Documentation and reports
│   ├── robust_optimization.md
│   └── sprawozdanie (2).pdf
│
└── plots/                     # Generated visualizations
```

## Modules

### 1. ETH AML Detector (`eth_aml/`)

Anti-money laundering wallet risk analysis combining two approaches:

**Rule-Based Detector** (`eth_aml_detector.py`)
- Seven hand-crafted heuristics: transaction cycles, fast transit, structuring, fan-out, fan-in, night activity, tainted links
- Builds a directed transaction graph from Etherscan data
- Generates a structured text report and network visualization

**LSTM Detector** (`eth_aml_lstm.py`)
- Bidirectional LSTM with self-attention pooling
- Trained on pseudo-labels from the rule-based scorer
- Extracts per-transaction attention weights for explainability
- Also fetches ERC-20 token transfer summaries

**Comparison** (`eth_aml_compare.py`)
- Runs both detectors on shared data (single API fetch)
- Merges results into `eth_aml_comparison.json`
- Provides a terminal diff table and accuracy vs expected verdicts

### 2. Crypto Explorer Pro (`crypto_explorer_pro/`)

Desktop application (Tkinter + Matplotlib) for cryptocurrency market analysis:

- Live price charts with MA(7), MA(25), Bollinger Bands, RSI(14)
- Interactive span selection for detailed sub-period analysis
- Multiple timeframes (24h to 1 year) and custom date ranges
- 12 supported cryptocurrencies via CoinGecko API
- Robust optimization tab for portfolio allocation under parameter uncertainty

### 3. Dashboard (`dashboard.html`)

Self-contained HTML page that visualizes the AML comparison results:
- Wallet risk cards with expandable breakdowns
- Rule-based vs LSTM score scatter plot
- Divergence analysis and verdict validation
- Token investment signals from on-chain data
- Dark theme, fully responsive

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd cryptotest

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your Etherscan API key
```

## Usage

### ETH AML Analysis

```bash
# Run the full comparison (fetches data, trains LSTM, writes JSON)
python eth_aml/eth_aml_compare.py

# Or run individually
python eth_aml/eth_aml_detector.py    # Rule-based only
python eth_aml/eth_aml_lstm.py         # LSTM only
```

Then open `dashboard.html` in a browser and load the generated `output/eth_aml_comparison.json`.

### Crypto Explorer Pro

```bash
cd crypto_explorer_pro
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.10+
- Etherscan API key (free tier available at [etherscan.io](https://etherscan.io/apis))
- No API key needed for Crypto Explorer Pro (uses free CoinGecko API)

## Dependencies

| Package | Purpose |
|---------|---------|
| `requests` | HTTP client for Etherscan and CoinGecko APIs |
| `pandas`, `numpy` | Data processing |
| `matplotlib`, `networkx` | Charting and graph visualization |
| `torch` | LSTM model training and inference |
| `scikit-learn` | Data preprocessing utilities |
| `python-dotenv` | Environment variable management |
