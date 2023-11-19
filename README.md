# Deep Neural Network Construction for Enhanced Multi-Asset Time Series and Cross-Sectional Momentum Strategies

## Abstract
> As one of the most applied portfolio strategies in practice, cross-sectional momentum (CSMOM) and time series momentum (TSMOM) serve as central concepts contradicting the efficient market hypothesis while obeying still unexplained market anomalies. While these strategies have been enhanced with deterministic model extensions with significant empirical out-performances over the last centuries, upcoming deep-learning approaches serve as a foundation for further optimizing the asset ranking for CSMOM and trend estimation as well as position sizing for TSMOM. This thesis focuses on deriving deep learning models from the ground up based on generalized CSMOM and TSMOM strategy frameworks as well as vanilla deep learning model architectures, with the goal of building momentum-specific deep learning models. Thereby, transformer neural networks serve as the foundation for the proposed \textit{CSMOM Transformer Ranker} and \textit{CSMOM ListNet Pre-Ranker Transformer Re-Ranker} based on the concept of self-attentive (re-)ranking by Pobrotyn et al. (2021), as well as the proposed sharpe-optimizing \textit{TSMOM Encoder-Decoder Transformer} and \textit{TSMOM Decoder-Only Transformer} based on empirical approaches on time series applications and the suggestion by Li et al. (2020) that the decoder-part might be sufficient for such applications. In addition, other deep learning models such as MLPs, RNNs, LSTMs, and abstractions of those, are utilized next to the original CSMOM and TSMOM models as benchmarks in order to empirically back-test the proposed transformer-based strategies on a multi-asset data set with 50 continuous future contracts between 2000 and 2022. The empirical back-testing results based on a monthly re-balancing frequency show that in the course of the CSMOM back-testing, the learning-to-rank (LTR) MLP and the deterministic MACD-based benchmark strategy with sharpe ratios of 1.0248 and 1.0007, respectively, dominate, while the proposed transformer-based models outperform the other benchmark strategies with sharpe ratios of 0.7997 and 0.6076, respectively. In the course of the TSMOM strategies, there is a global out-performance of the encoder-decoder transformer with a sharpe ratio of 0.8633, while the decoder-only transformer with a sharpe ratio of 0.4497 even lags behind other benchmark strategies, which in turn underlines the inherent design of the transformer encoder-decoder architecture for time series applications. In the course of comparison to existing research findings by Poh et al. (2022) and Wood et al. (2021), a likely reason for the mixed performance of the proposed transformer-based CSMOM and TSMOM models are the problems related to data scarcity, whereby due to the monthly re-balance frequency and the relatively small universe of 50 continuous futures contracts, there is no sufficiently large training set, which means the in-sample training cannot generalize which leads to an overfitting bias and was already observed based on the training, validation and testing losses.

## Setup
1. Download or clone the repository
2. Open the local terminal
3. Navigate into the project root directory
4. Create virtual environment by running: python3 -m venv venv"
5. Activate virtual envirionment by running:
   -- Mac/Linux: "source venv/bin/actrivate"
   -- Windows: "venv\Scripts\activate"
6. Install required packages by running "pip3 install -r requirements.txt"

## Run Back-Testing
1. Open the local terminal
2. Go through the steps of the setup guide above
3. Navigate into the project root directory
4. For CSMOM: run "python3 backtesting_csmom.py"; For TSMOM: run "python3 backtesting_tsmom.py"

## Key Results - CSMOM Back-Testing
![CSMOM Model Backtesting - Cumulative Returns](/figures_tables/CSMOM%20-%20Model%20Backtesting%20-%20Cumulative%20Returns.jpg)
![CSMOM Model Backtesting - Cumulative Returns (Log-Scale)](/figures_tables/CSMOM%20-%20Model%20Backtesting%20-%20Cumulative%20Returns%20(Log-Scale).jpg)

## Key Results - TSMOM Back-Testing
![TSMOM Model Backtesting - Cumulative Returns](/figures_tables/TSMOM%20-%20Model%20Backtesting%20-%20Cumulative%20Returns.jpg)
![TSMOM Model Backtesting - Cumulative Returns (Log-Scale)](/figures_tables/TSMOM%20-%20Model%20Backtesting%20-%20Cumulative%20Returns%20(Log-Scale).jpg)

## Full Thesis
![Thesis](thesis.pdf)