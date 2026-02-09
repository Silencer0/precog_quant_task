# 📊 Ultimate Quantitative Research & Backtesting Report

---

## 📅 Report Overview
This comprehensive report summarizes the entire quantitative research workflow implemented in this directory, spanning from raw data processing to advanced agentic portfolio management. The project is organized into four distinct tasks, each representing a critical stage in the alpha research pipeline.

---

## 🛠️ Task 1: Data Cleaning & Feature Extraction

### 1.1 Data Cleaning Methodology
The primary goal of the cleaning pipeline was to transform raw OHLCV stock data into "model-ready" stationary series while preserving the underlying statistical properties (e.g., jumps, volatility clusters).

**Key Steps Undertaken:**
1.  **Imputation**: Forward-filling missing values to maintain time-series continuity, followed by backward-filling for initial NaNs.
2.  **Integrity Validation**: Automated checks for negative prices, volume anomalies, and ticker consistency (see `research_quality_report.csv`).
3.  **Stationarity Transformation**: Conversion of raw prices $P_t$ to log-returns $r_t$ to ensure mean-reversion in historical windows.
4.  **No-Smoothing Baseline**: Crucially, the "Cleaned" dataset (`dataset/cleaned/`) avoids aggressive filtering to prevent leakage and "feature-smearing." Smoothing is instead implemented as *features*.

### 1.2 Feature Extraction Catalog
A robust set of **121 features** was extracted across multiple families.

| Family | Feature | Intuition / Math |
| :--- | :--- | :--- |
| **Returns** | `logret_1d` | $r_t = \ln(C_t / C_{t-1})$. Additive over time, convenient for statistical modeling. |
| **Momentum** | `rsi_14` | $RSI = 100 - \frac{100}{1+RS}$. Measures overbought/oversold regimes using Gains/Losses. |
| **Volatility** | `atr_14` | $ATR_t = \frac{1}{n} \sum TR_t$. $TR = \max(H-L, \|H-C_{t-1}\|, \|L-C_{t-1}\|)$. |
| **Volume** | `obv` | $OBV_t = OBV_{t-1} \pm V_t$. Accumulates volume based on price direction to detect distribution. |
| **Structure** | `bos` / `choch` | Break of Structure / Change of Character. Categorical features identifying trend reversals based on swing points. |
| **Spectral** | `kalman_filt` | 1D Kalman Filter state estimation ($x_t = x_{t-1} + w_t$). Extracts the 'invisible' denoised price. |

**Math Behind Spectral Features (Savitzky-Golay):**
We fit a degree-$d$ polynomial $p(i) = \sum_{j=0}^d a_j i^j$ to a data window. This provides a smoother derivative estimate than simple differencing, reducing noise in trend-following signals.

---

## 📈 Task 2: Backtesting Engine

### 2.1 Engine Architecture
The framework employs a dual-engine approach:
1.  **Vectorized Weights Engine**: High-speed simulation using NumPy/Pandas for bulk strategy testing.
2.  **Actor-Based Execution Engine** (`Backtester_Actor_Engine`): A higher-fidelity simulation utilizing stateful actors (`MemPortfolioActor`) to manage slots and liquidity constraints.

### 2.2 Visualization Layer
The backtesting suite produces interactive and static insights:
- **Equity/Drawdown Curves**: Standard line plots tracking capital growth and the 'underwater' percentage.
- **Weights Heatmap**: A temporal visualization of portfolio concentration, showing how the engine rotates capital between sectors or assets.
- **Pair Diagnostics**: Scatter plots of spread vs. z-score, illustrating entry/exit thresholds in cointegration models.

### 2.3 Core Quantitative Metrics
#### **Sharpe Ratio**
The Sharpe Ratio ($S$) measures excess return per unit of risk:
$$S = \frac{E[R_p - R_f]}{\sigma_p}$$
*   **Intuition:** It answers, "Is the return coming from smart management or just high volatility?" A Sharpe > 1.0 is generally considered good for daily strategies.

#### **Maximum Drawdown (The "Markdown" of Portfolios)**
Maximum Drawdown ($MDD$) represents the worst-case peak-to-trough decline:
$$MDD = \max_{\tau \in [0,T]} \left( \frac{P_{peak}(\tau) - P(\tau)}{P_{peak}(\tau)} \right)$$
*   **Intuition:** It measures the psychological and financial "pain" an investor must endure. A strategy with a 50% CAGR but 80% MDD is often untradable.

### 2.3 Portfolio Theory Approaches
*   **Modern Portfolio Theory (MPT)**: Optimization via Sharpe Maximization. We solve for weights $w$ that maximize expected return for a given level of variance.
*   **Neo-Modern Portfolio Theory (HRP)**: Hierarchical Risk Parity. Unlike MPT, which requires inverting a "noisy" covariance matrix, HRP uses tree-based clustering to allocate risk across uncorrelated asset groups.
*   **1/N Strategy**: Equal weighting (e.g., $w_i = 1/N$). Simple but robust; often beats MPT in out-of-sample tests due to lack of estimation error.

### 2.4 Agentic Behavior & Design
The "Agentic" backtester mimics a rational market participant:
- **State Management**: The engine maintains "Available Slots" (e.g., 20 maximum concurrent positions).
- **Macro-Micro Coupling**: Micro-models generate alpha, Macro-models provide "Risk-On/Off" scaling, and the Allocator (Portfolio Manager) enforces diversification.
- **Stop Loss Logic**: Discrete monitoring where if $P_{low} < P_{entry} \cdot (1 - SL)$, the position is force-closed at the next tick/open.

---

## 🤖 Task 3: Models & Results

### 3.1 Model Math & Theory
| Category | Model Theory | Technical Depth |
| :--- | :--- | :--- |
| **Bayesian** | Beta-Binomial | Uses conjugate priors to update the probability of a "up-move" daily: $P(\theta \| D) \propto P(D \| \theta)P(\theta)$. |
| **Boosted** | LightGBM / CatBoost | Iteratively builds decision trees to minimize a loss function (e.g., LogLoss) using gradients. Handles non-linear feature interactions. |
| **Deep Learning** | HGNN (Graph) | Models stocks as nodes in a graph where edges represent industry correlations. Propagates signals across linked assets. |
| **FinMamba** | State-Space Model | A Selective State Space Model architecture designed to handle long-range dependencies in time series without the $O(N^2)$ cost of Transformers. |

### 3.2 Performance Summary Table
Below is the synthesized performance data extracted from `ultimate_report_model_metrics.csv`.

| Model Design | CAGR [%] | Sharpe | Max Drawdown [%] |
| :--- | :--- | :--- | :--- |
| **Stochastic Volatility (Particle Filter)** | **27.73%** | **1.68** | **-12.85%** |
| **SMC Linear (Lasso) - 1/N** | 26.24% | 1.40 | -18.49% |
| **Bayes Logistic Regression (NUTS)** | 25.43% | 1.27 | -19.68% |
| **Random Forest Classifier (Time Split)** | 19.70% | 0.99 | -13.28% |
| **HGNN (Node+Relation) - Time Split** | 18.15% | 0.88 | -19.59% |
| **Ridge Regression - MPT** | 17.73% | 0.94 | -34.18% |
| **Ichimoku Strategy (1/N)** | 15.51% | 0.99 | -25.48% |
| **AdaBoost Classifier** | 12.57% | 0.77 | -15.92% |
| **Alpha 101 Ensemble (Mean)** | -10.75% | -2.11 | -67.58% |

---

## 🖇️ Task 4: Cointegration Research

### 4.1 Approaches & Methodology
Cointegration is the "holy grail" of pairs trading, identifying pairs that may wander apart but always return to a mean.

1.  **Engle-Granger Two-Step**: Regress Asset A on Asset B ($y = \beta x + \epsilon$) and test the residual $\epsilon$ for stationarity using the Augmented Dickey-Fuller (ADF) test.
2.  **K-Means on Slopes**: Clustering assets by their Heiken-Ashi slope correlations to find candidates for co-movement.
3.  **Kernel PCA Indicators**: Using non-linear dimensionality reduction to find common latent factors driving groups of stocks.

### 4.2 Intuition & Results
The intuition behind testing cointegration instead of simple correlation is that **correlation is a short-term co-movement**, while **cointegration is a long-term equilibrium**.
- **Result**: While simple technical strategies (RSI/MACD) performed well, the **RF Cointegration Pairs** model showed lower CAGR (2.07%) but extremely stable drawdown (-8.61%), proving its value as a capital preservation/low-beta strategy.

---

## 📚 Sources
*   *WorldQuant 101 Alphas*: Kakushadze, Z. (2016).
*   *FinMamba Architecture*: Deep learning for financial state-space models.
*   *Advanced Technical Indicators*: arXiv:2412.15448.
*   *Modern Portfolio Theory*: Markowitz (1952).

---
**Verified & Validated by Quant Research Engine**
