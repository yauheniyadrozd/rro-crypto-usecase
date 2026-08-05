# Raport: Optymalizacja Odporna (Robust Optimization) w Alokacji Aktywów Finansowych z Uwzględnieniem Kryptowalut

---

## Spis treści

1. [Wprowadzenie](#1-wprowadzenie)
2. [Kryteria akceptacji dla optymalizacji odpornej](#2-kryteria-akceptacji-dla-optymalizacji-odpornej)
3. [Modele optymalizacji odpornej](#3-modele-optymalizacji-odpornej)
4. [Miary ryzyka](#4-miary-ryzyka)
5. [Specyfika kryptowalut w optymalizacji odpornej](#5-specyfika-kryptowalut-w-optymalizacji-odpornej)
6. [Metodyka praktyczna: predykcja kursu, budżet, rebalancing](#6-metodyka-praktyczna-predykcja-kursu-budżet-rebalancing)
7. [Literatura i źródła](#7-literatura-i-źródła)
8. [Podsumowanie](#8-podsumowanie)

---

## 1. Wprowadzenie

### 1.1 Czym jest optymalizacja odporna?

Optymalizacja odporna (ang. *Robust Optimization*, RO) to podejście do optymalizacji portfelowej, które uwzględnia **niepewność parametrów wejściowych** — w szczególności estymowanych stóp zwrotu, macierzy kowariancji i kosztów transakcyjnych. W przeciwieństwie do klasycznej teorii Markowitza (1952), która zakłada, że parametry są znane z pewnością, RO konstruuje portfele, które są **odporne na błędy estymacji** i **scenariusze skrajne**.

**Intuicja:** Zamiast pytać *"jaki jest najlepszy portfel przy danych parametrach?"*, RO pyta *"jaki portfel będzie dobry nawet jeśli parametry okażą się błędne?"*

### 1.2 Dlaczego to ważne przy kryptowalutach?

Kryptowaluty charakteryzują się:
- **Ekstremalną zmiennością** (volatility rzędu 60–150% rocznie vs ~20% dla S&P 500)
- **Grubymi ogonami rozkładów** (zdarzenia typu *black swan* są częstsze)
- **Krótką historią danych** (BTC od 2009, większość altcoinów <5 lat)
- **Zmienną strukturą korelacji** (w czasie bessy korelacje gwałtownie rosną)
- **Ryzykiem regulacyjnym i technologicznym**

Te cechy sprawiają, że klasyczne estymatory parametrów są **skrajnie niepewne**, a RO jest naturalnym narzędziem do radzenia sobie z tą niepewnością.

---

## 2. Kryteria akceptacji dla optymalizacji odpornej

Poniżej przedstawiam zestaw **logicznych i łatwych do wdrożenia** kryteriów akceptacji, które można stosować przy budowie portfela z wykorzystaniem RO.

### 2.1 Kryteria podstawowe (logiczne, łatwe)

| # | Kryterium | Opis | Próg akceptacji |
|---|-----------|------|-----------------|
| **K1** | **Maksymalna waga pojedynczego aktywa** | Ogranicza koncentrację ryzyka | ≤ 20–30% portfela |
| **K2** | **Minimalna dywersyfikacja** | Liczba aktywów w portfelu | ≥ 5–8 aktywów |
| **K3** | **Maksymalny drawdown** | Największy obsunięcie kapitału w horyzoncie | ≤ 30–40% |
| **K4** | **Minimalny Sharpe ratio** | Zwrot na jednostkę ryzyka (w ujęciu odpornym) | ≥ 0.5 (dla pełnego portfela) |
| **K5** | **Maksymalna zmienność portfela** | Roczna zmienność (σ) portfela | ≤ 50–60% (dla portfeli z crypto) |
| **K6** | **Płynność** | Możliwość wyjścia z pozycji bez znaczącego poślizgu | ≥ 80% AUM do zlikwidowania w ≤ 24h |
| **K7** | **Odporność na stres-test** | Zachowanie portfela w scenariuszach skrajnych | CVaR₉₅ ≤ 15% straty dziennej |

### 2.2 Kryteria zaawansowane (z literatury)

| # | Kryterium | Opis | Źródło |
|---|-----------|------|--------|
| **K8** | **Worst-case expected return** | Minimalny oczekiwany zwrot w zbiorze niepewności | Ben-Tal & Nemirovski (1998) |
| **K9** | **Robust CVaR** | Conditional Value-at-Risk w najgorszym przypadku | Zhu & Fukushima (2009) |
| **K10** | **Stabilność wag** | Zmiana wag portfela przy małej perturbacji danych | DeMiguel et al. (2009) |
| **K11** | **Turnover constraint** | Ograniczenie kosztów transakcyjnych przy rebalancingu | ≤ 30–50% miesięcznie |
| **K12** | **Ellipsoidal uncertainty budget** | Parametr κ kontrolujący rozmiar zbioru niepewności | Fabozzi et al. (2007) |

### 2.3 Kryteria specyficzne dla crypto

| # | Kryterium | Opis |
|---|-----------|------|
| **K13** | **Market cap minimum** | Aktywo musi mieć kapitalizację ≥ $500M–$1B |
| **K14** | **Historia notowań** | Minimum 2–3 lata danych dziennych |
| **K15** | **Wolumen dzienny** | ≥ $10M daily volume (zabezpieczenie przed manipulacją) |
| **K16** | **Maksymalna ekspozycja na sektor** | Np. ≤ 40% w DeFi, ≤ 30% w L1, ≤ 20% w meme |
| **K17** | **Stablecoin buffer** | Minimum 5–10% w stablecoinach jako bufor płynności |

---

## 3. Modele optymalizacji odpornej

### 3.1 Model Markowitza (punkt odniesienia)

Klasyczny problem:
```
min  wᵀ Σ w
s.t. wᵀ μ ≥ r_target
     wᵀ 1 = 1
     w ≥ 0
```

**Problem:** μ i Σ są estymowane z danych → błąd estymacji → portfel niestabilny.

### 3.2 Robust Mean-Variance (najprostszy model odporny)

Wprowadzamy **zbiór niepewności** dla μ:

```
U_μ = { μ : ||μ - μ̂|| ≤ κ }
```

gdzie μ̂ to estymowana średnia, a κ kontroluje poziom odporności.

**Problem odporny:**
```
min  max_{μ ∈ U_μ, Σ ∈ U_Σ}  wᵀ Σ w
s.t. min_{μ ∈ U_μ} wᵀ μ ≥ r_target
     wᵀ 1 = 1, w ≥ 0
```

### 3.3 Robust Optimization z elipsoidalnym zbiorem niepewności

Najczęściej stosowany w literaturze (Ben-Tal & Nemirovski, 1999):

**Zbiór niepewności dla zwrotów:**
```
U = { r : r = r̂ + P·u, ||u||₂ ≤ κ }
```
gdzie P to macierz Cholesky'ego z Σ̂ (dekompozycja Σ̂ = P Pᵀ).

**Sformułowanie dla min-wariancji:**
```
min  wᵀ Σ̂ w  +  κ · ||Pᵀ w||₂
s.t. wᵀ μ̂ - κ · ||Pᵀ w||₂ ≥ r_target   (odporna wersja ograniczenia)
```

To prowadzi do programu **stożkowego drugiego rzędu (SOCP)** — rozwiązywalnego efektywnie.

### 3.4 Black-Litterman + Robust (rekomendowane dla crypto)

Model Blacka-Littermana (1992) pozwala łączyć:
- **Prior:** zwroty równowagowe (np. z CAPM lub równych wag)
- **Views:** subiektywne prognozy analityka

W wersji odpornej (Meucci, 2005; Schöttle et al., 2010):

```
μ_BL = [ (τΣ)⁻¹ + Pᵀ Ω⁻¹ P ]⁻¹ [ (τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q ]
```

gdzie:
- Π = zwroty równowagowe (np. z market-cap weights)
- P = macierz pick (które aktywa mają *view*)
- Q = wektor oczekiwanych zwrotów z *views*
- Ω = macierz niepewności *views*
- τ = parametr skalujący niepewność priora

**Zaleta dla crypto:** Można wstrzyknąć fundamentalne *views* (np. "BTC będzie rósł względem ETH") nie polegając wyłącznie na historycznych danych.

### 3.5 Worst-Case CVaR (Zhu & Fukushima, 2009)

Zamiast wariancji, optymalizujemy CVaR w najgorszym przypadku:

```
min  max_{p ∈ P}  CVaR_α(w, p)
s.t. wᵀ 1 = 1, w ≥ 0
```

gdzie P to zbiór możliwych rozkładów (np. mieszanina rozkładów dopuszczających grube ogony).

---

## 4. Miary ryzyka

### 4.1 Miary klasyczne

| Miara | Wzór / Opis | Zalety | Wady |
|-------|-------------|--------|------|
| **Wariancja / σ** | σ² = wᵀ Σ w | Prosta, dobrze zbadana | Symetryczna, nie rozróżnia zysków od strat |
| **VaR₉₅** | Kwantyl 5% rozkładu strat | Regulacyjny standard | Nie subaddytywny, ignoruje kształt ogona |
| **CVaR₉₅** | Średnia strata powyżej VaR₉₅ | Subaddytywny, lepiej opisuje ogon | Wymaga więcej danych |
| **Max Drawdown** | MDD = max(peak - trough) / peak | Intuicyjny, popularny w crypto | Zależny od ścieżki |

### 4.2 Miary odporne (rekomendowane)

| Miara | Opis |
|-------|------|
| **Robust CVaR** | CVaR w najgorszym przypadku ze zbioru niepewności P |
| **Worst-case variance** | Maksymalna wariancja przy perturbacji Σ w zbiorze U |
| **Entropic Risk Measure** | ρ(X) = (1/θ) log E[exp(-θX)], wrażliwy na ogon |
| **Expectiles** | Uogólnienie kwantyli, elastyczne ważenie strat |

### 4.3 Miary ryzyka dla crypto (dodatkowe)

| Miara | Opis |
|-------|------|
| **Tail dependence** | Korelacja w ogonie — czy aktywa spadają razem w kryzysie? |
| **Liquidity risk** | Spread bid-ask, slippage przy dużych zleceniach |
| **Protocol risk** | Ryzyko smart-contract, hacku, regulacji (trudno kwantyfikowalne) |
| **Concentration risk** | % supply w top-10 adresach (whale risk) |

---

## 5. Specyfika kryptowalut w optymalizacji odpornej

### 5.1 Wyzwania

1. **Krótka historia** → estymatory Σ i μ mają wysoką wariancję → potrzeba RO
2. **Grube ogony** → VaR i CVaR niedoszacowują ryzyka → **skorygowany CVaR z rozkładami Pareto**
3. **Zmienne korelacje** → w bessie korelacja BTC-ETH rośnie z 0.5 do >0.9 → **dynamiczna macierz kowariancji** (np. DCC-GARCH)
4. **Asymetria informacji** → insider trading, whale manipulation → **shrinkage estimator** dla Σ
5. **Staking / yield** → dodatkowy zwrot z tytułu stakingu → uwzględnić w μ z korektą o ryzyko slashingu

### 5.2 Proponowane podejście hybrydowe

```
Portfel = Core (BTC, ETH) + Satellites (altcoiny) + Buffer (stablecoiny)
```

| Warstwa | Udział | Aktywa | Rebalancing |
|---------|--------|--------|-------------|
| **Core** | 40–60% | BTC, ETH | Rzadki (kwartalnie) |
| **Satellites** | 20–40% | Top-10 altcoinów | Częstszy (miesięcznie) |
| **Buffer** | 5–20% | USDC, USDT, DAI | Wg potrzeb |

### 5.3 Estymatory odporne dla krypto-danych

**Shrinkage Ledoit-Wolf (2004):**
```
Σ_shrink = δ · Σ_target + (1-δ) · Σ_sample
```
gdzie Σ_target to np. macierz stałych korelacji lub single-factor model.
δ jest estymowane optymalnie z danych.

**Wartość δ dla krypto:** typowo δ ∈ [0.3, 0.6] — większy shrinkage niż dla akcji (δ ~ 0.2), co odzwierciedla większą niepewność estymacji.

---

## 6. Metodyka praktyczna: predykcja kursu, budżet, rebalancing

### 6.1 Schemat procesu (end-to-end)

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Predykcja    │ →  │ Optymalizacja     │ →  │ Alokacja         │
│ kursów       │    │ odporna          │    │ (budżet B)       │
│ (μ̂, Σ̂, views)│    │ (wagi w*)        │    │ (ilości n_i)     │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │ Monitoring +     │
                                             │ Rebalancing      │
                                             │ (trigger-based)  │
                                             └─────────────────┘
```

### 6.2 Krok 1: Predykcja kursów

Dla każdego aktywa i:

**a) Model średniej (trend):**
- **Prosty:** Średnia krocząca (30-dniowa, 90-dniowa)
- **Średni:** Regresja liniowa z momentum (factor: 1M, 3M, 6M momentum)
- **Zaawansowany:** ARIMA/GARCH dla zmienności + regresja dla trendu

**b) Model zmienności i korelacji:**
- EWMA (λ = 0.94, standard RiskMetrics)
- DCC-GARCH (dynamic conditional correlation)
- Shrinkage Ledoit-Wolf (odporny)

**c) Views (Black-Litterman):**
- "BTC będzie outperformance ETH o 10% w ciągu 6 miesięcy"
- "Sektor DeFi wzrośnie szybciej niż L1"
- "Zmienność wzrośnie w Q4"

### 6.3 Krok 2: Optymalizacja odporna

Dla budżetu **B** (np. $10,000):

```python
# Pseudokod optymalizacji odpornej
min  wᵀ Σ_shrink w  +  κ · ||Pᵀ w||₂
s.t.
    wᵀ μ_BL  -  κ · ||Pᵀ w||₂  ≥  r_target     # odporny zwrot
    wᵀ 1 = 1                                     # pełna alokacja
    0 ≤ w_i ≤ w_max                              # max waga
    Σ_i w_i · 𝟙(sector_i = s) ≤ sector_max_s     # limit sektorowy
    Σ_i w_i · 𝟙(liquidity_i < threshold) ≤ 0.1   # limit illiquid
    turnover ≤ 0.3                                # max zmiana wag
```

**Kalibracja κ (budżet niepewności):**

| Poziom odporności | κ     | Interpretacja |
|-------------------|-------|---------------|
| Niski (agresywny) | 0–0.5 | Mała niepewność — portfel bliski Markowitzowi |
| Średni            | 0.5–1.5 | Zrównoważona odporność |
| Wysoki (konserw.) | 1.5–3.0 | Duża niepewność — portfel bliski equal-weight |
| Crypto-adapted    | 1.0–2.5 | Rekomendowany zakres dla portfeli z crypto |

### 6.4 Krok 3: Alokacja (zamiana wag na ilości)

Dla budżetu B i cen p_i:

```
n_i = (w_i · B) / p_i    (zaokrąglone do standardowego lota)
```

**Przykład z $10,000:**

| Aktywo | Waga w_i | Alokacja $ | Cena | Ilość |
|--------|----------|------------|------|-------|
| BTC    | 30%      | $3,000     | $60,000 | 0.05 BTC |
| ETH    | 25%      | $2,500     | $3,000 | 0.833 ETH |
| SOL    | 15%      | $1,500     | $150 | 10 SOL |
| LINK   | 10%      | $1,000     | $15 | 66.7 LINK |
| USDC   | 10%      | $1,000     | $1.00 | 1,000 USDC |
| AVAX   | 10%      | $1,000     | $35 | 28.6 AVAX |

### 6.5 Krok 4: Rebalancing

**Reguły triggerowe (proste i logiczne):**

| Trigger | Akcja | Częstotliwość |
|---------|-------|---------------|
| Waga odchyliła się > ±5pp od celu | Rebalansuj do celu | Check dzienny |
| Zmienność BTC > 100% annualized | Zmniejsz ekspozycję o 20% | Check dzienny |
| Korelacja crypto-coin > 0.9 | Zwiększ wagę stablecoinów | Check tygodniowy |
| Kalendarzowy | Pełny rebalancing | Miesięcznie |
| Nowy sygnał predykcyjny | Aktualizuj μ_BL i re-optymalizuj | Na żądanie |

**Formuła rebalancingu z kosztami transakcyjnymi:**

```
w_new = argmin [ (w - w_target)ᵀ Σ (w - w_target)  +  cᵀ |w - w_current| ]
```

gdzie c to wektor kosztów transakcyjnych (0.1–0.5% dla crypto na CEX).

### 6.6 Monitoring ryzyka (dashboard)

| Metryka | Próg ostrzegawczy | Próg krytyczny |
|---------|-------------------|-----------------|
| Portfolio σ (30d) | > 50% | > 80% |
| Max drawdown | > 20% | > 35% |
| CVaR₉₅ (dzienny) | > 5% | > 10% |
| Korelacja BTC-ETH (30d) | > 0.8 | > 0.95 |
| % w stablecoinach | < 5% | < 2% |
| Koncentracja top-3 | > 60% | > 80% |

---

## 7. Literatura i źródła

### 7.1 Klasyczne prace o optymalizacji odpornej

| Autor(zy) | Rok | Tytuł | Kluczowy wkład |
|-----------|-----|-------|-----------------|
| **Markowitz, H.** | 1952 | *Portfolio Selection*, Journal of Finance | Teoria średniej-wariancji (punkt wyjścia) |
| **Ben-Tal, A. & Nemirovski, A.** | 1998 | *Robust Convex Optimization*, Mathematics of Operations Research | Teoretyczne podstawy RO — zbiory niepewności |
| **Ben-Tal, A. & Nemirovski, A.** | 1999 | *Robust solutions of uncertain linear programs*, Operations Research Letters | Elipsoidalne zbiory niepewności |
| **Goldfarb, D. & Iyengar, G.** | 2003 | *Robust Portfolio Selection Problems*, Mathematics of Operations Research | RO dla portfeli — niepewność μ i Σ |
| **Tütüncü, R.H. & Koenig, M.** | 2004 | *Robust Asset Allocation*, Annals of Operations Research | RO z kosztami transakcyjnymi |

### 7.2 Estymatory odporne i shrinkage

| Autor(zy) | Rok | Tytuł | Kluczowy wkład |
|-----------|-----|-------|-----------------|
| **Ledoit, O. & Wolf, M.** | 2004 | *A well-conditioned estimator for large-dimensional covariance matrices*, Journal of Multivariate Analysis | Shrinkage estymator macierzy kowariancji |
| **Ledoit, O. & Wolf, M.** | 2003 | *Improved estimation of the covariance matrix of stock returns*, Journal of Empirical Finance | Shrinkage dla małego T / dużego N |
| **DeMiguel, V., Garlappi, L. & Uppal, R.** | 2009 | *Optimal Versus Naive Diversification*, Review of Financial Studies | 1/N często bije "optymalne" — argument za RO |

### 7.3 Black-Litterman i rozszerzenia

| Autor(zy) | Rok | Tytuł | Kluczowy wkład |
|-----------|-----|-------|-----------------|
| **Black, F. & Litterman, R.** | 1992 | *Global Portfolio Optimization*, Financial Analysts Journal | Model BL — łączenie priora z views |
| **Meucci, A.** | 2005 | *Risk and Asset Allocation*, Springer | Kompleksowe podejście, BL + copula + RO |
| **Schöttle, K., Werner, R. & Zagst, R.** | 2010 | *Comparison and robustification of Bayes and Black-Litterman models*, Mathematical Methods of OR | Robust BL |

### 7.4 CVaR i miary ryzyka

| Autor(zy) | Rok | Tytuł | Kluczowy wkład |
|-----------|-----|-------|-----------------|
| **Rockafellar, R.T. & Uryasev, S.** | 2000 | *Optimization of Conditional Value-at-Risk*, Journal of Risk | CVaR — definicja i optymalizacja |
| **Zhu, S. & Fukushima, M.** | 2009 | *Worst-Case Conditional Value-at-Risk*, Operations Research | Robust CVaR — CVaR w zbiorze niepewności |
| **Fabozzi, F.J., Kolm, P.N., Pachamanova, D.A. & Focardi, S.M.** | 2007 | *Robust Portfolio Optimization and Management*, Wiley | Kompleksowy podręcznik RO w finansach |

### 7.5 Crypto i aktywa cyfrowe

| Autor(zy) | Rok | Tytuł | Kluczowy wkład |
|-----------|-----|-------|-----------------|
| **Corbet, S. et al.** | 2018 | *Cryptocurrencies as a financial asset*, International Review of Financial Analysis | Charakterystyka crypto jako klasy aktywów |
| **Chuen, D.L.K., Guo, L. & Wang, Y.** | 2017 | *Cryptocurrency: A New Investment Opportunity?*, Journal of Alternative Investments | Crypto w portfelu — dywersyfikacja |
| **Petukhina, A., Trimborn, S., Härdle, W.K. & Elendner, H.** | 2020 | *Investing with cryptocurrencies*, Digital Finance | Optymalizacja portfela z crypto |
| **Trimborn, S. & Härdle, W.K.** | 2018 | *CRIX an Index for cryptocurrencies*, Journal of Empirical Finance | Indeksowanie rynku crypto |

### 7.6 Dynamiczne i zaawansowane

| Autor(zy) | Rok | Tytuł | Kluczowy wkład |
|-----------|-----|-------|-----------------|
| **Engle, R.** | 2002 | *Dynamic Conditional Correlation*, Journal of Business & Economic Statistics | DCC-GARCH — dynamiczne korelacje |
| **Kim, W.C., Kim, J.H. & Fabozzi, F.J.** | 2016 | *Robust Equity Portfolio Management*, Wiley | RO + formuła na κ optymalne |
| **Scherer, B.** | 2004 | *Portfolio Construction and Risk Budgeting*, Risk Books | Risk budgeting i RO w praktyce |

---

## 8. Podsumowanie

### 8.1 Kluczowe wnioski

1. **Optymalizacja odporna jest naturalnym wyborem dla portfeli z kryptowalutami** ze względu na ekstremalną niepewność parametrów wynikającą z krótkiej historii, grubych ogonów i zmiennych korelacji.

2. **Najprostsze i najskuteczniejsze kryteria akceptacji** to:
   - Maksymalna waga pojedynczego aktywa (≤ 20–30%)
   - CVaR₉₅ w najgorszym przypadku (≤ 15% dziennie)
   - Minimalny bufor stablecoinów (≥ 5–10%)
   - Market cap i wolumen jako filtr jakości aktywów

3. **Rekomendowane podejście hybrydowe:**
   - **Predykcja:** Shrinkage Ledoit-Wolf dla Σ + Black-Litterman dla μ z subiektywnymi views
   - **Optymalizacja:** Robust Mean-Variance z elipsoidalnym zbiorem niepewności (SOCP)
   - **Struktura:** Core (BTC, ETH) + Satellites (top altcoiny) + Buffer (stablecoiny)
   - **Rebalancing:** Trigger-based (daily check) + kalendarzowy (miesięcznie), z kosztami transakcyjnymi

4. **Parametr κ** (budżet niepewności) powinien być w zakresie **1.0–2.5** dla portfeli crypto — wyższy niż dla tradycyjnych akcji (0.5–1.0).

### 8.2 Ścieżka wdrożenia (MVP)

```
Tydzień 1: Zbieranie danych — ceny daily BTC, ETH, top-20 altcoinów (3 lata)
Tydzień 2: Implementacja estymatorów (Σ_shrink, μ_BL)
Tydzień 3: Implementacja solvera RO (SOCP, np. cvxpy w Pythonie)
Tydzień 4: Backtest + kalibracja κ + definicja kryteriów akceptacji
Tydzień 5: Wdrożenie z małym budżetem (np. $1,000) + monitoring
```

### 8.3 Ryzyka i ograniczenia

| Ryzyko | Mitigacja |
|--------|-----------|
| Model nie przewidzi *black swana* | CVaR + stress-testy z ekstremalnymi scenariuszami |
| Korelacje rosną w kryzysie | DCC-GARCH + dynamiczny κ |
| Koszty transakcyjne zjadają zyski | Turnover constraint + trigger-based rebalancing |
| Ryzyko giełdy (CEX upada) | Dywersyfikacja po giełdach lub self-custody |
| Regulacje zakazują crypto | Buffer stablecoinów + limit ekspozycji |
| Overfitting do danych historycznych | RO + shrinkage + out-of-sample validation |

---

*Raport przygotowany: sierpień 2026. Wszystkie modele i kryteria należy zwalidować na danych out-of-sample przed wdrożeniem na rzeczywistym kapitale.*
