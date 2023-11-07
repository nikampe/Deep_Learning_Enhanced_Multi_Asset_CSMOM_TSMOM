import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameters
from settings.hyper_params import hyperparams_fixed_csmom, hyperparams_grid_csmom

class LambdaMART:
    def __init__(self, data_csmom):
        # Load CSMOM data
        self.X_arr_train, self.y_arr_train, \
        self.X_arr_valid, self.y_arr_valid, \
        self.X_arr_test, self.y_arr_test, \
        self.X_indices_arr_train, self.y_indices_arr_train, \
        self.X_indices_arr_valid, self.y_indices_arr_valid, \
        self.X_indices_arr_test, self.y_indices_arr_test = data_csmom

    def train(self, X_train, y_train, X_valid, y_valid):
        # Combine Train and Validation Data
        X_train = np.vstack((X_train, X_valid))
        y_train = np.vstack((y_train, y_valid))
        # Model Building
        model = xgb.XGBRegressor(
            objective=hyperparams_fixed_csmom["LambdaMART"]['objective'], 
            eval_metric = hyperparams_fixed_csmom["LambdaMART"]['eval_metric'],
            verbose = 3)
        # Hyperparameter Tuning
        model_tuned = RandomizedSearchCV(
            model,
            param_distributions=hyperparams_grid_csmom["LambdaMART"], 
            n_iter = hyperparams_fixed_csmom["LambdaMART"]['random_search_trials'], 
            scoring = 'neg_mean_squared_error')
        # Model Training
        model_tuned_fitted = model_tuned.fit(X_train, y_train)
        return model_tuned_fitted

    def weights(self, n):
        # Initiate Predictions Array
        weights = []
        # Loop over Training & Prediction Intervals
        num_intervals = len(self.X_arr_train)
        for interval in range(num_intervals):
            # Weight Initialization
            weights_interval = np.zeros(shape=(self.y_arr_test[interval].shape[0], self.y_arr_test[interval].shape[1]))
            # Ground-Truth Decile Labels
            deciles_train = np.zeros_like(self.y_arr_train[interval])
            for x in range(self.y_arr_train[interval].shape[0]):
                daily_returns = self.y_arr_train[interval][x, :, 0]
                decile_thresholds = [np.percentile(daily_returns, t * 10) for t in range(1, 10)]
                daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                deciles_train[x, :, 0] = daily_deciles
            deciles_valid = np.zeros_like(self.y_arr_valid[interval])
            for x in range(self.y_arr_valid[interval].shape[0]):
                daily_returns = self.y_arr_valid[interval][x, :, 0]
                decile_thresholds = [np.percentile(daily_returns, t * 10) for t in range(1, 10)]
                daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                deciles_valid[x, :, 0] = daily_deciles
            # Model Training
            model_tuned_fitted = self.train(
                self.X_arr_train[interval].reshape(self.X_arr_train[interval].shape[0]*self.X_arr_train[interval].shape[1], self.X_arr_train[interval].shape[2]),
                deciles_train.reshape(deciles_train.shape[0]*deciles_train.shape[1], 1), 
                self.X_arr_valid[interval].reshape(self.X_arr_valid[interval].shape[0]*self.X_arr_valid[interval].shape[1], self.X_arr_valid[interval].shape[2]), 
                deciles_valid.reshape(deciles_valid.shape[0]*deciles_valid.shape[1], 1))
            # Model Testing
            pred = model_tuned_fitted.predict(
                self.X_arr_test[interval].reshape(self.X_arr_test[interval].shape[0]*self.X_arr_test[interval].shape[1], self.X_arr_test[interval].shape[2]))
            pred = pred.reshape(self.y_arr_test[interval].shape[0], self.y_arr_test[interval].shape[1])
            # Extract Top and Bottom Assets
            top_n_assets = np.argsort(pred, axis=1)[:, -int(n/2):]
            bottom_n_assets = np.argsort(pred, axis=1)[:, :int(n/2)]
            # Update Weight Predictions
            for q, top_n_assets_i in enumerate(top_n_assets):
                weights_interval[q][top_n_assets_i] = 1
            for q, bottom_n_assets_i in enumerate(bottom_n_assets):
                weights_interval[q][bottom_n_assets_i] = -1
            weights = np.vstack([weights, weights_interval]) if len(weights) > 0 else weights_interval
        np.savetxt(f"data/predictions/csmom_lambdamart_weights.csv", weights, delimiter=",")
        return weights