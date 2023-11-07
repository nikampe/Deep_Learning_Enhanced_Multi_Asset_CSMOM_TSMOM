import numpy as np

class MACD:
    def __init__(self, prices, params, rebalance_days):
        self.prices = prices
        self.params = params
        self.rebalance_days = rebalance_days

    def macd(self):
        macds = []
        for span_short, span_long in zip(self.params["macd_span_short"], self.params["macd_span_long"]):
            halflife_short = np.log(0.5) / np.log(1 - 1 / span_short)
            halflife_long = np.log(0.5) / np.log(1 - 1 / span_long)
            macd_indicator_short = self.prices.ewm(halflife=halflife_short).mean()
            macd_indicator_long = self.prices.ewm(halflife=halflife_long).mean()
            macd_indicator = macd_indicator_short - macd_indicator_long
            macd_indicator = macd_indicator / self.prices.rolling(window=self.params["macd_std_short"]).std()
            macd_indicator = macd_indicator / self.prices.rolling(window=self.params["macd_std_long"]).std()
            macds.append(macd_indicator)
        macd_combined = 1/len(macds) * (macds[0] + macds[1] + macds[2])
        return macd_combined

    def weights(self, n):
        n_top, n_bottom = int(n/2), int(n/2)
        macd_combined = self.macd()
        macd_combined = macd_combined[macd_combined.index.isin(self.rebalance_days)]
        weights = macd_combined.to_numpy()
        for weights_daily in weights:
            largest_indices = np.argpartition(weights_daily, -n_top)[-n_top:]
            smallest_indices = np.argpartition(weights_daily, n_bottom)[:n_bottom]
            weights_daily[largest_indices] = 1
            weights_daily[smallest_indices] = -1
            weights_daily[np.setdiff1d(np.arange(weights_daily.size), np.concatenate((largest_indices, smallest_indices)))] = 0
        np.savetxt(f"data/predictions/csmom_macd_weights.csv", weights, delimiter=",")
        return weights