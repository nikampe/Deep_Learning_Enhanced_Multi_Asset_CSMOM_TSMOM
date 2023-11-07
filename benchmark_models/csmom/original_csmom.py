import numpy as np

class CSMOM:
    def __init__(self, returns, rebalance_days):
        self.returns = returns
        self.rebalance_days = rebalance_days

    def weights(self, lookback, n):
        n_top, n_bottom = int(n/2), int(n/2)
        if lookback > 1:
            cumulative_returns = self.returns.rolling(lookback).apply(lambda x: (1+x).prod()-1) 
        else:
            cumulative_returns = self.returns.copy()
        cumulative_returns = cumulative_returns[cumulative_returns.index.isin(self.rebalance_days)]
        weights = cumulative_returns.to_numpy()
        for weights_daily in weights:
            largest_indices = np.argpartition(weights_daily, -n_top)[-n_top:]
            smallest_indices = np.argpartition(weights_daily, n_bottom)[:n_bottom]
            weights_daily[largest_indices] = 1 # 1/n
            weights_daily[smallest_indices] = -1 # 1/n
            weights_daily[np.setdiff1d(np.arange(weights_daily.size), np.concatenate((largest_indices, smallest_indices)))] = 0
        np.savetxt(f"data/predictions/csmom_original_{lookback}m_weights.csv", weights, delimiter=",")
        return weights