import numpy as np

class TSMOM:
    def __init__(self, returns, rebalance_days):
        self.returns = returns
        self.rebalance_days = rebalance_days

    def weights(self, lookback):
        cumulative_returns = self.returns.rolling(lookback).apply(lambda x: (1+x).prod()-1) 
        cumulative_returns.dropna(how="any", axis=0)
        cumulative_returns = cumulative_returns[cumulative_returns.index.isin(self.rebalance_days)]
        weights = cumulative_returns.applymap(np.sign)
        weights = weights.to_numpy()
        np.savetxt(f"data/predictions/tsmom_original_{int(lookback/21)}m_weights.csv", weights, delimiter=",")
        return weights