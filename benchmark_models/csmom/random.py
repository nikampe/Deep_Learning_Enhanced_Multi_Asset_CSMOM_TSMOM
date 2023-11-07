import numpy as np

class CSMOMRandom:
    def __init__(self, num_days, num_assets):
        self.num_days = num_days
        self.num_assets = num_assets

    def weights(self, n):
        weights = np.zeros((self.num_days, self.num_assets))
        for i in range(self.num_days):
            ones_indices = np.random.choice(self.num_assets, int(n/2), replace=False)
            minus_ones_indices = np.random.choice(list(set(range(self.num_assets)) - set(ones_indices)), int(n/2), replace=False)
            weights[i, ones_indices] = 1
            weights[i, minus_ones_indices] = -1
        np.savetxt(f"data/predictions/csmom_random_weights.csv", weights, delimiter=",")
        return weights