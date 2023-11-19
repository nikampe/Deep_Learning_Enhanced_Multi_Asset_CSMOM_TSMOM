import numpy as np

class TSMOMRandom:
    def __init__(self, num_days, num_assets):
        self.num_days = num_days
        self.num_assets = num_assets

    def weights(self):
        weights = np.random.choice([-1, 1], size=(self.num_days, self.num_assets))
        np.savetxt(f"data/predictions/tsmom_random_weights.csv", weights, delimiter=",")
        return weights