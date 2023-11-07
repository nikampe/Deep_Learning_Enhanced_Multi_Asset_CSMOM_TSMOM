import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from empyrical import sharpe_ratio, calmar_ratio, sortino_ratio, max_drawdown, downside_risk, annual_return, annual_volatility

# Global variables
DAYS_PER_MONTH = 21
MONTHS_PER_YEAR = 12

def returns(df):
    for col in df.columns:
        df[col] = df[col].pct_change()
    return df

def log_returns(df):
    for col in df.columns:
        df[col] = np.log(df[col]) - np.log(df[col].shift(1))
    return df

def cumulative_returns(df_returns, cumulative_periods):
    cumulative_returns = []
    for period in cumulative_periods:
        cumulative_returns_period = df_returns.rolling(period).apply(lambda x: (1+x).prod()-1) 
        cumulative_returns.append(cumulative_returns_period)
    return cumulative_returns

def normalized_returns(df_returns, normalized_periods, vol_lookback, vol_target):
    normalized_returns = []
    for period in normalized_periods:
        cumulative_returns_period = df_returns.rolling(period).apply(lambda x: (1+x).prod()-1) 
        vol_daily = df_returns.ewm(span=vol_lookback, min_periods=vol_lookback).std().fillna(method="bfill")
        normalized_returns_period = cumulative_returns_period / vol_daily / np.sqrt(period)
        normalized_returns.append(normalized_returns_period)
    return normalized_returns

def indicator_csmom(returns, indicator_periods, n=10):
    indicators_csmom = []
    for period in indicator_periods:
        cumulative_returns = (returns+1).rolling(period).apply(np.prod)-1
        df_transformed = cumulative_returns.apply(indicator_csmom_per_row, axis=1, n=n)
        indicators_csmom.append(df_transformed)
    return indicators_csmom

def indicator_csmom_per_row(row, n):
    if row.isna().all():
        return row
    largest_values = row.nlargest(n).index
    smallest_values = row.nsmallest(n).index
    new_row = pd.Series(0, index=row.index)
    new_row[largest_values] = 1
    new_row[smallest_values] = -1
    return new_row

def macd_csmom(df_prices, params):
    df_macd = []
    for span_short, span_long in zip(params["macd_span_short"], params["macd_span_long"]):
        halflife_short = np.log(0.5) / np.log(1 - 1 / span_short)
        halflife_long = np.log(0.5) / np.log(1 - 1 / span_long)
        macd_indicator = df_prices.ewm(halflife=halflife_short).mean() - df_prices.ewm(halflife=halflife_long).mean()
        macd_indicator = macd_indicator / df_prices.rolling(window=params["macd_std_short"]).std().fillna(method="bfill")
        macd_indicator = macd_indicator / df_prices.rolling(window=params["macd_std_long"]).std().fillna(method="bfill")
        df_macd.append(macd_indicator)
    return df_macd

def macd_tsmom(df_prices, params):
    df_macd = []
    for span_short, span_long in zip(params["macd_span_short"], params["macd_span_long"]):
        halflife_short = np.log(0.5) / np.log(1 - 1 / span_short)
        halflife_long = np.log(0.5) / np.log(1 - 1 / span_long)
        macd_indicator = df_prices.ewm(halflife=halflife_short).mean() - df_prices.ewm(halflife=halflife_long).mean()
        macd_indicator = macd_indicator / df_prices.rolling(window=params["macd_std_short"]).std().fillna(method="bfill")
        macd_indicator = macd_indicator / df_prices.rolling(window=params["macd_std_long"]).std().fillna(method="bfill")
        df_macd.append(macd_indicator)
    return df_macd

def asset_class(data_returns, data_assets, map):
    df_asset_class = pd.DataFrame(index=data_returns.index, columns=data_returns.columns)
    for col in df_asset_class.columns:
        ac = data_assets.loc[col, "Asset Class"]
        df_asset_class.loc[:,col] = map[ac]
    return df_asset_class

def clean_data(df_features):
    df_features = df_features.fillna(method="ffill").dropna(axis=0, how="any")
    return df_features

def csmom_train_valid_test_split(dfs, df_returns, train_interval, train_valid_split, rebalance_freq, vol_lookback, vol_target, preloaded_data=False):
    if preloaded_data:
        X_arr_train = np.array(np.load("data/preloaded/data_csmom_X_arr_train.npy", allow_pickle=True))
        y_arr_train = np.array(np.load("data/preloaded/data_csmom_y_arr_train.npy", allow_pickle=True))
        X_arr_valid = np.array(np.load("data/preloaded/data_csmom_X_arr_valid.npy", allow_pickle=True))
        y_arr_valid = np.array(np.load("data/preloaded/data_csmom_y_arr_valid.npy", allow_pickle=True))
        X_arr_test = np.array(np.load("data/preloaded/data_csmom_X_arr_test.npy", allow_pickle=True))
        y_arr_test = np.array(np.load("data/preloaded/data_csmom_y_arr_test.npy", allow_pickle=True))
        X_arr_indices_train = np.load("data/preloaded/data_csmom_X_arr_indices_train.npy", allow_pickle=True)
        y_arr_indices_train = np.load("data/preloaded/data_csmom_y_arr_indices_train.npy", allow_pickle=True)
        X_arr_indices_valid = np.load("data/preloaded/data_csmom_X_arr_indices_valid.npy", allow_pickle=True)
        y_arr_indices_valid = np.load("data/preloaded/data_csmom_y_arr_indices_valid.npy", allow_pickle=True)
        X_arr_indices_test = np.load("data/preloaded/data_csmom_X_arr_indices_test.npy", allow_pickle=True)
        y_arr_indices_test = np.load("data/preloaded/data_csmom_y_arr_indices_test.npy", allow_pickle=True)
    else:
        # Adjust for Rebalance Frequency, Concatenate and Clean Features
        dfs = [df.iloc[np.arange(0, df.shape[0], rebalance_freq)] for df in dfs]
        dfs_concatenated = clean_data(pd.concat(dfs, axis=1))
        num_assets = len(dfs[0].columns)
        column_names = []
        indicator = 1
        for num_column, column_name in enumerate(dfs_concatenated.columns):
            column_names.append(column_name+"0"+str(indicator)) if indicator < 10 else column_names.append(column_name+str(indicator))
            if (num_column+1) % num_assets == 0:
                indicator += 1
        dfs_concatenated.columns = column_names
        dfs_concatenated.sort_index(axis=1, inplace=True)
        # Cumulative returns
        if rebalance_freq > 1:
            df_returns_cumulative = df_returns.rolling(window=rebalance_freq).apply(lambda x: (1+x).prod()-1)
        else:
            df_returns_cumulative = df_returns.copy()
        df_returns_cumulative = df_returns_cumulative[df_returns_cumulative.index.isin(dfs_concatenated.index)]
        df_returns_cumulative.sort_index(axis=1, inplace=True)
        # Volatility returns
        df_volatility = df_returns.ewm(span=vol_lookback, min_periods=vol_lookback).std().fillna(method="bfill")
        df_volatility = df_volatility.shift(1) * np.sqrt(252)
        # Train, Validation, Test Interval Split Indices
        num_splits = len(dfs_concatenated) // train_interval + 1
        split_indices = [i*train_interval for i in range(num_splits)]
        split_indices.append(len(dfs_concatenated))
        # Train-Validation-Test Interval Splits
        num_assets = len(dfs[0].columns)
        num_features = len(dfs)
        X_arr_train, X_arr_valid, X_arr_test = [], [], []
        y_arr_train, y_arr_valid, y_arr_test = [], [], []
        X_arr_indices_train, X_arr_indices_valid, X_arr_indices_test = [], [], []
        y_arr_indices_train, y_arr_indices_valid, y_arr_indices_test = [], [], []
        ## Train-Validation Interval Splits
        for i in range(0, len(split_indices)-2):
            # Train-Validation Split Index
            start_train, end_train = split_indices[i], split_indices[i+1] - int((split_indices[i+1]-split_indices[i])*(1-train_valid_split))
            start_valid, end_valid = split_indices[i+1] - int((split_indices[i+1]-split_indices[i])*(1-train_valid_split)), split_indices[i+1]
            df_train = dfs_concatenated.iloc[start_train:end_train]
            df_valid = dfs_concatenated.iloc[start_valid:end_valid-1]
            # Dimension Parameters
            num_indices_train = len(df_train.index)
            num_indices_valid = len(df_valid.index)
            # Train-Validation Input
            X_train = df_train.values.reshape((num_indices_train, num_assets, num_features))
            X_valid = df_valid.values.reshape((num_indices_valid, num_assets, num_features))
            X_arr_train.append(X_train)
            X_arr_valid.append(X_valid)
            # Train-Validation Output
            # y_train = df_returns_cumulative.iloc[start_train+1:end_train+1].values
            # y_valid = df_returns_cumulative.iloc[start_valid+1:end_valid].values
            y_indices_train = df_returns_cumulative.iloc[start_train+1:end_train+1].index
            y_indices_valid = df_returns_cumulative.iloc[start_valid+1:end_valid].index
            y_train = (vol_target / df_volatility.loc[df_train.index,:].values) * df_returns_cumulative.loc[y_indices_train,:].values
            y_valid = (vol_target / df_volatility.loc[df_valid.index,:].values) * df_returns_cumulative.loc[y_indices_valid,:].values
            y_arr_train.append(y_train.reshape((num_indices_train, num_assets, 1)))
            y_arr_valid.append(y_valid.reshape((num_indices_valid, num_assets, 1)))
            # Train-Validation Rebalance Days
            X_arr_indices_train.append(df_train.index)
            X_arr_indices_valid.append(df_valid.index)
            y_arr_indices_train.append(y_indices_train) # (y_train.index)
            y_arr_indices_valid.append(y_indices_valid) # (y_valid.index)
        ## Test Interval Splits
        for i in range(1, len(split_indices)-1):
            # Test Split Index
            start_test, end_test = split_indices[i], split_indices[i+1]
            df_test = dfs_concatenated.iloc[start_test:end_test-1]
            # Dimension parameters
            num_indices_test = len(df_test.index)
            # Test Input
            X_test = df_test.values.reshape((num_indices_test, num_assets, num_features))
            X_arr_test.append(X_test)
            # Test Output
            # y_test = df_returns_cumulative.iloc[start_test+1:end_test].values
            y_indices_test = df_returns_cumulative.iloc[start_test+1:end_test].index
            y_test = (vol_target / df_volatility.loc[df_test.index,:].values) * df_returns_cumulative.loc[y_indices_test,:].values
            y_arr_test.append(y_test.reshape((num_indices_test, num_assets, 1)))
            # Test Rebalance Days
            X_arr_indices_test.append(df_test.index)
            y_arr_indices_test.append(y_indices_test)
        # # Save Train-Test-Validation Sets
        # np.save("data/preloaded/data_csmom_X_arr_train.npy", np.array(X_arr_train))
        # np.save("data/preloaded/data_csmom_y_arr_train.npy", np.array(y_arr_train))
        # np.save("data/preloaded/data_csmom_X_arr_valid.npy", np.array(X_arr_valid))
        # np.save("data/preloaded/data_csmom_y_arr_valid.npy", np.array(y_arr_valid))
        # np.save("data/preloaded/data_csmom_X_arr_test.npy", np.array(X_arr_test))
        # np.save("data/preloaded/data_csmom_y_arr_test.npy", np.array(y_arr_test))
        # np.save("data/preloaded/data_csmom_X_arr_indices_train.npy", X_arr_indices_train)
        # np.save("data/preloaded/data_csmom_y_arr_indices_train.npy", y_arr_indices_train)
        # np.save("data/preloaded/data_csmom_X_arr_indices_valid.npy", X_arr_indices_valid)
        # np.save("data/preloaded/data_csmom_y_arr_indices_valid.npy", y_arr_indices_valid)
        # np.save("data/preloaded/data_csmom_X_arr_indices_test.npy", X_arr_indices_test)
        # np.save("data/preloaded/data_csmom_y_arr_indices_test.npy", y_arr_indices_test)
    # Print Data Set Summary
    summary_cols = [f"Batch #{i+1}" for i in range(0, len(X_arr_train))]
    summary = pd.DataFrame(index=["Train", "Validation", "Test"], columns=summary_cols)
    for i in range(0, len(X_arr_train)):
        summary.loc["Train", summary_cols[i]] = str(X_arr_indices_train[i][0])[5:7] + "/" + str(X_arr_indices_train[i][0])[0:4] + " - " + str(X_arr_indices_train[i][-1])[5:7] + "/" + str(X_arr_indices_train[i][-1])[0:4] 
        summary.loc["Validation", summary_cols[i]] = str(X_arr_indices_valid[i][0])[5:7] + "/" + str(X_arr_indices_valid[i][0])[0:4] + " - " + str(X_arr_indices_valid[i][-1])[5:7] + "/" + str(X_arr_indices_valid[i][-1])[0:4]
        summary.loc["Test", summary_cols[i]] = str(X_arr_indices_test[i][0])[5:7] + "/" + str(X_arr_indices_test[i][0])[0:4] + " - " + str(X_arr_indices_test[i][-1])[5:7] + "/" + str(X_arr_indices_test[i][-1])[0:4]
    print("\nCSMOM Train-Validation-Split Interval Summary:\n", summary)
    summary.to_latex("figures_tables/Tables/CSMOM - Input Data Dates - Transformer Ranker.txt")
    summary.to_latex("figures_tables/Tables/CSMOM - Input Data Dates - Transformer Re-  Ranker.txt")
    # Print Data Set Shape Summary
    summary_shapes = pd.DataFrame(index=["Train", "Validation", "Test"], columns=summary_cols)
    for i in range(0, len(X_arr_train)):
        summary_shapes.loc["Train", summary_cols[i]] = "X: " + str(np.array(X_arr_train[i]).shape) + " | " + "y: " + str(np.array(y_arr_train[i]).shape)
        summary_shapes.loc["Validation", summary_cols[i]] = "X: " + str(np.array(X_arr_valid[i]).shape) + " | " + "y: " + str(np.array(y_arr_valid[i]).shape)
        summary_shapes.loc["Test", summary_cols[i]] = "X: " + str(np.array(X_arr_test[i]).shape) + " | " + "y: " + str(np.array(y_arr_test[i]).shape)
    print("\nCSMOM Train-Validation-Split Interval Shape Summary:\n", summary_shapes)
    summary_shapes.to_latex("figures_tables/Tables/CSMOM - Input Data Shapes - Transformer Ranker.txt")
    summary_shapes.to_latex("figures_tables/Tables/CSMOM - Input Data Shapes - Transformer Re-Ranker.txt")
    return (X_arr_train, y_arr_train, X_arr_valid, y_arr_valid, X_arr_test, y_arr_test, X_arr_indices_train, y_arr_indices_train, X_arr_indices_valid, y_arr_indices_valid, X_arr_indices_test, y_arr_indices_test)

def tsmom_train_valid_test_split(lookback, dfs, df_returns, train_valid_split, rebalance_freq, vol_lookback, vol_target, preloaded_data=False):
    if preloaded_data:
        X_arr_train = np.array(np.load("data/preloaded/data_csmom_X_arr_train.npy", allow_pickle=True))
        y_arr_train = np.array(np.load("data/preloaded/data_csmom_y_arr_train.npy", allow_pickle=True))
        X_arr_valid = np.array(np.load("data/preloaded/data_csmom_X_arr_valid.npy", allow_pickle=True))
        y_arr_valid = np.array(np.load("data/preloaded/data_csmom_y_arr_valid.npy", allow_pickle=True))
        X_arr_test = np.array(np.load("data/preloaded/data_csmom_X_arr_test.npy", allow_pickle=True))
        y_arr_test = np.array(np.load("data/preloaded/data_csmom_y_arr_test.npy", allow_pickle=True))
        X_arr_indices_train = np.array(np.load("data/preloaded/data_csmom_X_arr_indices_train.npy", allow_pickle=True))
        y_arr_indices_train = np.array(np.load("data/preloaded/data_csmom_y_arr_indices_train.npy", allow_pickle=True))
        X_arr_indices_valid = np.array(np.load("data/preloaded/data_csmom_X_arr_indices_valid.npy", allow_pickle=True))
        y_arr_indices_valid = np.array(np.load("data/preloaded/data_csmom_y_arr_indices_valid.npy", allow_pickle=True))
        X_arr_indices_test = np.array(np.load("data/preloaded/data_csmom_X_arr_indices_test.npy", allow_pickle=True))
        y_arr_indices_test = np.array(np.load("data/preloaded/data_csmom_y_arr_indices_test.npy", allow_pickle=True))
        ##########################
        X_enc_arr_train = np.load("data/preloaded/data_csmom_X_enc_arr_train.npy", allow_pickle=True)
        X_dec_arr_train = np.load("data/preloaded/data_csmom_X_dec_arr_train.npy", allow_pickle=True)
        y_arr_train = np.load("data/preloaded/data_csmom_y_arr_train.npy", allow_pickle=True)
        X_enc_arr_valid = np.load("data/preloaded/data_csmom_X_enc_arr_valid.npy", allow_pickle=True)
        X_dec_arr_valid = np.load("data/preloaded/data_csmom_X_dec_arr_valid.npy", allow_pickle=True)
        y_arr_valid = np.load("data/preloaded/data_csmom_y_arr_valid.npy", allow_pickle=True)
        X_enc_arr_test = np.load("data/preloaded/data_csmom_X_enc_arr_test.npy", allow_pickle=True)
        X_dec_arr_test = np.load("data/preloaded/data_csmom_X_dec_arr_test.npy", allow_pickle=True)
        y_arr_test = np.load("data/preloaded/data_csmom_y_arr_test.npy", allow_pickle=True)
        X_indices_arr_train = np.load("data/preloaded/data_csmom_X_indices_arr_train.npy", allow_pickle=True)
        y_indices_arr_train = np.load("data/preloaded/data_csmom_y_indices_arr_train.npy", allow_pickle=True)
        X_indices_arr_valid = np.load("data/preloaded/data_csmom_X_indices_arr_valid.npy", allow_pickle=True)
        y_indices_arr_valid = np.load("data/preloaded/data_csmom_y_indices_arr_valid.npy", allow_pickle=True)
        X_indices_arr_test = np.load("data/preloaded/data_csmom_X_indices_arr_test.npy", allow_pickle=True)
        y_indices_arr_test = np.load("data/preloaded/data_csmom_y_indices_arr_test.npy", allow_pickle=True)
        ##########################
    else:
        # Concatenate and clean features
        dfs = [df.iloc[np.arange(0, df.shape[0], rebalance_freq)] for df in dfs]
        dfs_concatenated = clean_data(pd.concat(dfs, axis=1))
        num_assets = len(dfs[0].columns)
        column_names = []
        indicator = 1
        for num_column, column_name in enumerate(dfs_concatenated.columns):
            column_names.append(column_name+"0"+str(indicator)) if indicator < 10 else column_names.append(column_name+str(indicator))
            if (num_column+1) % num_assets == 0:
                indicator += 1
        dfs_concatenated.columns = column_names
        dfs_concatenated.sort_index(axis=1, inplace=True)
        # Cumulative returns
        if rebalance_freq > 1:
            df_returns_cumulative = df_returns.rolling(window=rebalance_freq).apply(lambda x: (1+x).prod()-1)
        else:
            df_returns_cumulative = df_returns.copy()
        df_returns_cumulative = df_returns_cumulative[df_returns_cumulative.index.isin(dfs_concatenated.index)]
        df_returns_cumulative.sort_index(axis=1, inplace=True) 
        # Volatility returns
        df_volatility = df_returns.ewm(span=vol_lookback, min_periods=vol_lookback).std().fillna(method="bfill")
        # Dimension parameters
        assets = df_returns_cumulative.columns
        num_assets = len(df_returns_cumulative.columns)
        num_features = len(dfs)
        # Train, Validation, Test Batch Indices
        num_batches = int(len(dfs_concatenated)-1) // int(60)
        train_indices, valid_indices, test_indices = [], [], []
        for i in range(1, num_batches + 1):
            end_idx = int(len(dfs_concatenated)-1) - (i-1) * int(60)
            start_idx = end_idx - int(60)
            if i != 1:
                train_indices.append([start_idx, end_idx]) # 0-48
            if i != num_batches:
                test_indices.append([start_idx, end_idx]) # 0-60
        train_indices = train_indices[::-1]
        valid_indices = valid_indices[::-1]
        for i in range(0, len(train_indices)):
            train_indices[i][0] = train_indices[0][0]
            valid_indices.append([train_indices[i][1]-((i+1)*lookback), train_indices[i][1]]) # 48-60
            train_indices[i][1] = train_indices[i][1]-(i+1)*lookback
        test_indices = test_indices[::-1]
        ##########################
        X_enc_arr_train, X_dec_arr_train, X_enc_dec_indices_arr_train = [], [], []
        X_enc_arr_valid, X_dec_arr_valid, X_enc_dec_indices_arr_valid = [], [], []
        X_enc_arr_test, X_dec_arr_test, X_enc_dec_indices_arr_test = [], [], []
        y_dec_arr_train, y_dec_indices_arr_train = [], []
        y_dec_arr_valid, y_dec_indices_arr_valid = [], []
        y_dec_arr_test, y_dec_indices_arr_test = [], []
        ##########################
        # Train-Validation-Test Interval Splits
        X_arr_train, X_arr_valid, X_arr_test = [], [], []
        y_arr_train, y_arr_valid, y_arr_test = [], [], []
        X_arr_indices_train, X_arr_indices_valid, X_arr_indices_test = [], [], []
        y_arr_indices_train, y_arr_indices_valid, y_arr_indices_test = [], [], []
        # Train-Validation Interval Splits
        for m in range(0, len(train_indices)):
            # Train Interval Split
            start_train, end_train = train_indices[m][0], train_indices[m][1]
            df_X_train = dfs_concatenated.iloc[start_train:end_train]
            df_y_train = df_returns_cumulative.iloc[start_train+1:end_train+1]
            df_X_enc_dec_train = dfs_concatenated.iloc[start_train-lookback+1:end_train]
            df_y_dec_train = df_returns_cumulative.iloc[start_train-lookback+2:end_train+1]
            # Append Indices
            X_index_train = df_X_train.index
            y_index_train = df_y_train.index
            X_enc_dec_index_train = df_X_enc_dec_train.index
            y_dec_index_train = df_y_dec_train.index
            # Extract Rebalance Dates
            X_index_interval_train = [X_index_train.to_list()[i:i+lookback] for i in range(0, len(X_index_train.to_list()), lookback)]
            y_index_interval_train = [y_index_train.to_list()[i:i+lookback] for i in range(0, len(y_index_train.to_list()), lookback)]
            X_enc_dec_index_interval_train = [X_enc_dec_index_train.to_list()[i:i+lookback] for i in range(0, len(X_enc_dec_index_train.to_list())-lookback+1)]
            y_dec_index_interval_train = [y_dec_index_train.to_list()[i:i+lookback] for i in range(0, len(y_dec_index_train.to_list())-lookback+1)]
            # Volatility Scaling
            df_y_train = (vol_target / df_volatility.loc[y_index_train,:]) * df_returns_cumulative.loc[y_index_train,:]
            # Train Input & Output
            X_train, y_train = [], []
            for j in range(0, len(X_index_interval_train)):
                for asset in assets:
                    asset_features_i = [asset_feature_i for asset_feature_i in df_X_train.columns if asset_feature_i.startswith(asset)]
                    X_train_i = df_X_train.loc[X_index_interval_train[j], asset_features_i].values
                    y_train_i = df_y_train.loc[y_index_interval_train[j], asset].values
                    # Append to Array
                    X_train.append(X_train_i)
                    y_train.append(y_train_i)
            X_enc_dec_train, y_dec_train = [], []
            for j in range(0, len(X_enc_dec_index_interval_train)):
                for asset in assets:
                    asset_features_i = [asset_feature_i for asset_feature_i in df_X_train.columns if asset_feature_i.startswith(asset)]
                    X_enc_dec_train_i = df_X_enc_dec_train.loc[X_enc_dec_index_interval_train[j], asset_features_i].values
                    y_dec_train_i = df_y_dec_train.loc[y_dec_index_interval_train[j], asset].values
                    # Append to Array
                    X_enc_dec_train.append(X_enc_dec_train_i)
                    y_dec_train.append(y_dec_train_i)
            # Transform to Array and Reshape (Decoder-Only Structure)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
            X_arr_train.append(X_train)
            y_arr_train.append(y_train)
            X_arr_indices_train.append(pd.DatetimeIndex(X_index_train))
            y_arr_indices_train.append(pd.DatetimeIndex(y_index_train))
            # Transform to Array and Reshape (Encoder-Decoder Structure)
            X_enc_dec_train = np.array(X_enc_dec_train)
            y_dec_train = np.array(y_dec_train)
            X_enc_train = X_enc_dec_train[:, :X_enc_dec_train.shape[1]-1, :]
            X_dec_train = X_enc_dec_train[:, -1:, :]
            y_dec_train = y_dec_train.reshape(y_dec_train.shape[0], y_dec_train.shape[1], 1)
            y_dec_train = y_dec_train[:, -1:, :]
            X_enc_arr_train.append(X_enc_train)
            X_dec_arr_train.append(X_dec_train)
            y_dec_arr_train.append(y_dec_train)
            X_enc_dec_indices_arr_train.append(pd.DatetimeIndex(X_index_train))
            y_dec_indices_arr_train.append(pd.DatetimeIndex(y_index_train))
        # Validation Interval Split
        for m in range(0, len(valid_indices)):
            start_valid, end_valid = valid_indices[m][0], valid_indices[m][1]
            df_X_valid = dfs_concatenated.iloc[start_valid:end_valid]
            df_y_valid = df_returns_cumulative.iloc[start_valid+1:end_valid+1]
            df_X_enc_dec_valid = dfs_concatenated.iloc[start_valid-lookback+1:end_valid]
            df_y_dec_valid = df_returns_cumulative.iloc[start_valid-lookback+2:end_valid+1]
            # Append Indices
            X_index_valid = df_X_valid.index
            y_index_valid = df_y_valid.index
            X_enc_dec_index_valid = df_X_enc_dec_valid.index
            y_dec_index_valid = df_y_dec_valid.index
            # Extract Rebalance Dates
            X_index_interval_valid = [X_index_valid.to_list()[i:i+lookback] for i in range(0, len(X_index_valid.to_list()), lookback)]
            y_index_interval_valid = [y_index_valid.to_list()[i:i+lookback] for i in range(0, len(y_index_valid.to_list()), lookback)]
            X_enc_dec_index_interval_valid = [X_enc_dec_index_valid.to_list()[i:i+lookback] for i in range(0, len(X_enc_dec_index_valid.to_list())-lookback+1)]
            y_dec_index_interval_valid = [y_dec_index_valid.to_list()[i:i+lookback] for i in range(0, len(y_dec_index_valid.to_list())-lookback+1)]
            # Volatility Scaling
            df_y_valid = (vol_target / df_volatility.loc[y_index_valid,:]) * df_returns_cumulative.loc[y_index_valid,:]
            # Train Input & Output
            X_valid, y_valid = [], []
            for j in range(0, len(X_index_interval_valid)):
                for asset in assets:
                    asset_features_i = [asset_feature_i for asset_feature_i in df_X_valid.columns if asset_feature_i.startswith(asset)]
                    X_valid_i = df_X_valid.loc[X_index_interval_valid[j], asset_features_i].values
                    y_valid_i = df_y_valid.loc[y_index_interval_valid[j], asset].values
                    # Append to Array
                    X_valid.append(X_valid_i)
                    y_valid.append(y_valid_i)
            X_enc_dec_valid, y_dec_valid = [], []
            for j in range(0, len(X_enc_dec_index_interval_valid)):
                for asset in assets:
                    asset_features_i = [asset_feature_i for asset_feature_i in df_X_valid.columns if asset_feature_i.startswith(asset)]
                    X_enc_dec_valid_i = df_X_enc_dec_valid.loc[X_enc_dec_index_interval_valid[j], asset_features_i].values
                    y_dec_valid_i = df_y_dec_valid.loc[y_dec_index_interval_valid[j], asset].values
                    # Append to Array
                    X_enc_dec_valid.append(X_enc_dec_valid_i)
                    y_dec_valid.append(y_dec_valid_i)
            # Transform to Array and Reshape (Decoder-Only Structure)
            X_valid = np.array(X_valid)
            y_valid = np.array(y_valid)
            y_valid = y_valid.reshape(y_valid.shape[0], y_valid.shape[1], 1)
            X_arr_valid.append(X_valid)
            y_arr_valid.append(y_valid)
            X_arr_indices_valid.append(pd.DatetimeIndex(X_index_valid))
            y_arr_indices_valid.append(pd.DatetimeIndex(y_index_valid))
            # Transform to Array and Reshape (Encoder-Decoder Structure)
            X_enc_dec_valid = np.array(X_enc_dec_valid)
            y_dec_valid = np.array(y_dec_valid)
            X_enc_valid = X_enc_dec_valid[:, :X_enc_dec_valid.shape[1]-1, :]
            X_dec_valid = X_enc_dec_valid[:, -1:, :]
            y_dec_valid = y_dec_valid.reshape(y_dec_valid.shape[0], y_dec_valid.shape[1], 1)
            y_dec_valid = y_dec_valid[:, -1:, :]
            X_enc_arr_valid.append(X_enc_valid)
            X_dec_arr_valid.append(X_dec_valid)
            y_dec_arr_valid.append(y_dec_valid)
            X_enc_dec_indices_arr_valid.append(pd.DatetimeIndex(X_index_valid))
            y_dec_indices_arr_valid.append(pd.DatetimeIndex(y_index_valid))
        # Test Interval Splits
        for m in range(0, len(valid_indices)):
            start_test, end_test = test_indices[m][0], test_indices[m][1]
            df_X_test = dfs_concatenated.iloc[start_test:end_test]
            df_y_test = dfs_concatenated.iloc[start_test+1:end_test+1]
            df_X_enc_dec_test = dfs_concatenated.iloc[start_test-lookback+1:end_test]
            df_y_dec_test = df_returns_cumulative.iloc[start_test-lookback+2:end_test+1]
            # Append Indices
            X_index_test = df_X_test.index
            y_index_test = df_y_test.index
            X_enc_dec_index_test= df_X_enc_dec_test.index
            y_dec_index_test = df_y_dec_test.index
            # Extract Rebalance Dates
            X_index_interval_test = [X_index_test.to_list()[i:i+lookback] for i in range(0, len(X_index_test.to_list()), lookback)]
            y_index_interval_test = [y_index_test.to_list()[i:i+lookback] for i in range(0, len(y_index_test.to_list()), lookback)]
            X_enc_dec_index_interval_test = [X_enc_dec_index_test.to_list()[i:i+lookback] for i in range(0, len(X_enc_dec_index_test.to_list())-lookback+1)]
            y_dec_index_interval_test = [y_dec_index_test.to_list()[i:i+lookback] for i in range(0, len(y_dec_index_test.to_list())-lookback+1)]
            # Volatility Scaling
            df_y_test= (vol_target / df_volatility.loc[y_index_test,:]) * df_returns_cumulative.loc[y_index_test,:]
            # Train Input & Output
            X_test, y_test = [], []
            for j in range(0, len(X_index_interval_test)):
                for asset in assets:
                    asset_features_i = [asset_feature_i for asset_feature_i in df_X_test.columns if asset_feature_i.startswith(asset)]
                    X_test_i = df_X_test.loc[X_index_interval_test[j], asset_features_i].values
                    y_test_i = df_y_test.loc[y_index_interval_test[j], asset].values
                    # Append to Array
                    X_test.append(X_test_i)
                    y_test.append(y_test_i)
            X_enc_dec_test, y_dec_test = [], []
            for j in range(0, len(X_enc_dec_index_interval_test)):
                for asset in assets:
                    asset_features_i = [asset_feature_i for asset_feature_i in df_X_test.columns if asset_feature_i.startswith(asset)]           
                    X_enc_dec_test_i = df_X_enc_dec_test.loc[X_enc_dec_index_interval_test[j], asset_features_i].values
                    y_dec_test_i = df_y_dec_test.loc[y_dec_index_interval_test[j], asset].values
                    # Append to Array
                    X_enc_dec_test.append(X_enc_dec_test_i)
                    y_dec_test.append(y_dec_test_i)
            # Transform to Array and Reshape
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
            X_arr_test.append(X_test)
            y_arr_test.append(y_test)
            X_arr_indices_test.append(pd.DatetimeIndex(X_index_test))
            y_arr_indices_test.append(pd.DatetimeIndex(y_index_test))
            # Transform to Array and Reshape (Encoder-Decoder Structure)
            X_enc_dec_test = np.array(X_enc_dec_test)
            y_dec_test = np.array(y_dec_test)
            X_enc_test = X_enc_dec_test[:, :X_enc_dec_test.shape[1]-1, :]
            X_dec_test = X_enc_dec_test[:, -1:, :]
            y_dec_test = y_dec_test.reshape(y_dec_test.shape[0], y_dec_test.shape[1], 1)
            y_dec_test = y_dec_test[:, -1:, :]
            X_enc_arr_test.append(X_enc_test)
            X_dec_arr_test.append(X_dec_test)
            y_dec_arr_test.append(y_dec_test)
            X_enc_dec_indices_arr_test.append(pd.DatetimeIndex(X_index_test))
            y_dec_indices_arr_test.append(pd.DatetimeIndex(y_index_test))
        # np.save("data/preloaded/data_csmom_X_arr_train.npy", np.array(X_arr_train))
        # np.save("data/preloaded/data_csmom_y_arr_train.npy", np.array(y_arr_train))
        # np.save("data/preloaded/data_csmom_X_arr_valid.npy", np.array(X_arr_valid))
        # np.save("data/preloaded/data_csmom_y_arr_valid.npy", np.array(y_arr_valid))
        # np.save("data/preloaded/data_csmom_X_arr_test.npy", np.array(X_arr_test))
        # np.save("data/preloaded/data_csmom_y_arr_test.npy", np.array(y_arr_test))
        # np.save("data/preloaded/data_csmom_X_arr_indices_train.npy", X_arr_indices_train)
        # np.save("data/preloaded/data_csmom_y_arr_indices_train.npy", y_arr_indices_train)
        # np.save("data/preloaded/data_csmom_X_arr_indices_valid.npy", X_arr_indices_valid)
        # np.save("data/preloaded/data_csmom_y_arr_indices_valid.npy", y_arr_indices_valid)
        # np.save("data/preloaded/data_csmom_X_arr_indices_test.npy", X_arr_indices_test)
        # np.save("data/preloaded/data_csmom_y_arr_indices_test.npy", y_arr_indices_test)
        ##########################
        # np.save("data/preloaded/data_csmom_X_enc_arr_train.npy", np.array(X_enc_arr_train))
        # np.save("data/preloaded/data_csmom_X_dec_arr_train.npy", np.array(X_dec_arr_train))
        # np.save("data/preloaded/data_csmom_y_arr_train.npy", np.array(y_arr_train))
        # np.save("data/preloaded/data_csmom_X_enc_arr_valid.npy", np.array(X_enc_arr_valid))
        # np.save("data/preloaded/data_csmom_X_dec_arr_valid.npy", np.array(X_dec_arr_valid))
        # np.save("data/preloaded/data_csmom_y_arr_valid.npy", np.array(y_arr_valid))
        # np.save("data/preloaded/data_csmom_X_enc_arr_test.npy", np.array(X_enc_arr_test))
        # np.save("data/preloaded/data_csmom_X_dec_arr_test.npy", np.array(X_dec_arr_test))
        # np.save("data/preloaded/data_csmom_y_arr_test.npy", np.array(y_arr_test))
        # np.save("data/preloaded/data_csmom_X_indices_arr_train.npy", X_indices_arr_train)
        # np.save("data/preloaded/data_csmom_y_indices_arr_train.npy", y_indices_arr_train)
        # np.save("data/preloaded/data_csmom_X_indices_arr_valid.npy", X_indices_arr_valid)
        # np.save("data/preloaded/data_csmom_y_indices_arr_valid.npy", y_indices_arr_valid)
        # np.save("data/preloaded/data_csmom_X_indices_arr_test.npy", X_indices_arr_test)
        # np.save("data/preloaded/data_csmom_y_indices_arr_test.npy", y_indices_arr_test)
        ##########################
    ##########################
    # Print Data Set Summary
    summary_cols = [f"Batch #{i+1}" for i in range(0, len(X_enc_arr_train))]
    summary = pd.DataFrame(index=["Train", "Validation", "Test"], columns=summary_cols)
    for i in range(0, len(X_enc_arr_train)):
        summary.loc["Train", summary_cols[i]] = str(X_enc_dec_indices_arr_train[i][0])[5:7] + "/" + str(X_enc_dec_indices_arr_train[i][0])[0:4] + " - " + str(X_enc_dec_indices_arr_train[i][-1])[5:7] + "/" + str(X_enc_dec_indices_arr_train[i][-1])[0:4] 
        summary.loc["Validation", summary_cols[i]] = str(X_enc_dec_indices_arr_valid[i][0])[5:7] + "/" + str(X_enc_dec_indices_arr_valid[i][0])[0:4] + " - " + str(X_enc_dec_indices_arr_valid[i][-1])[5:7] + "/" + str(X_enc_dec_indices_arr_valid[i][-1])[0:4]
        summary.loc["Test", summary_cols[i]] = str(X_enc_dec_indices_arr_test[i][0])[5:7] + "/" + str(X_enc_dec_indices_arr_test[i][0])[0:4] + " - " + str(X_enc_dec_indices_arr_test[i][-1])[5:7] + "/" + str(X_enc_dec_indices_arr_test[i][-1])[0:4]
    print("\nTSMOM Encoder-Decoder Train-Validation-Split Interval Summary:\n", summary)
    summary.to_latex("figures_tables/Tables/TSMOM - Input Data Dates - Encoder-Decoder Transformer.txt")
    # Print Data Set Shape Summary
    summary_shapes = pd.DataFrame(index=["Train", "Validation", "Test"], columns=summary_cols)
    for i in range(0, len(X_enc_arr_train)):
        summary_shapes.loc["Train", summary_cols[i]] =  "X (Enc): " + str(np.array(X_enc_arr_train[i]).shape) + " , " + "X (Dec): " + str(np.array(X_dec_arr_train[i]).shape) + " , " + "y: " + str(np.array(y_dec_arr_train[i]).shape)
        summary_shapes.loc["Validation", summary_cols[i]] = "X (Enc):  " + str(np.array(X_enc_arr_valid[i]).shape) + " , " + "X (Dec):  " + str(np.array(X_dec_arr_valid[i]).shape) + " , " + "y:  " + str(np.array(y_dec_arr_valid[i]).shape)
        summary_shapes.loc["Test", summary_cols[i]] = "X (Enc): " + str(np.array(X_enc_arr_test[i]).shape) + " , " + "X (Dec): " + str(np.array(X_dec_arr_test[i]).shape) + " , " + "y: " + str(np.array(y_dec_arr_test[i]).shape)
    print("\nTSMOM Encoder-Decoder Train-Validation-Split Interval Shape Summary:\n", summary_shapes)
    summary_shapes.to_latex("figures_tables/Tables/TSMOM - Input Data Shapes - Encoder-Decoder Transformer.txt")
    ##########################
    # Print Data Set Summary
    summary_cols = [f"Batch #{i+1}" for i in range(0, len(X_arr_train))]
    summary = pd.DataFrame(index=["Train", "Validation", "Test"], columns=summary_cols)
    for i in range(0, len(X_arr_train)):
        summary.loc["Train", summary_cols[i]] = str(X_arr_indices_train[i][0])[5:7] + "/" + str(X_arr_indices_train[i][0])[0:4] + " - " + str(X_arr_indices_train[i][-1])[5:7] + "/" + str(X_arr_indices_train[i][-1])[0:4] 
        summary.loc["Validation", summary_cols[i]] = str(X_arr_indices_valid[i][0])[5:7] + "/" + str(X_arr_indices_valid[i][0])[0:4] + " - " + str(X_arr_indices_valid[i][-1])[5:7] + "/" + str(X_arr_indices_valid[i][-1])[0:4]
        summary.loc["Test", summary_cols[i]] = str(X_arr_indices_test[i][0])[5:7] + "/" + str(X_arr_indices_test[i][0])[0:4] + " - " + str(X_arr_indices_test[i][-1])[5:7] + "/" + str(X_arr_indices_test[i][-1])[0:4]
    print("\nTSMOM Decoder-Only Train-Validation-Split Interval Summary:\n", summary)
    summary.to_latex("figures_tables/Tables/TSMOM - Input Data Dates - Decoder Transformer.txt")
    # Print Data Set Shape Summary
    summary_shapes = pd.DataFrame(index=["Train", "Validation", "Test"], columns=summary_cols)
    for i in range(0, len(X_arr_train)):
        summary_shapes.loc["Train", summary_cols[i]] =  "X: " + str(np.array(X_arr_train[i]).shape) + " , " + "y: " + str(np.array(y_arr_train[i]).shape)
        summary_shapes.loc["Validation", summary_cols[i]] = "X:  " + str(np.array(X_arr_valid[i]).shape) + " , " + "y:  " + str(np.array(y_arr_valid[i]).shape)
        summary_shapes.loc["Test", summary_cols[i]] = "X: " + str(np.array(X_arr_test[i]).shape) + " , " + "y: " + str(np.array(y_arr_test[i]).shape)
    print("\nTSMOM Decoder-Only Train-Validation-Split Interval Shape Summary:\n", summary_shapes)
    summary_shapes.to_latex("figures_tables/Tables/TSMOM - Input Data Shapes - Decoder Transformer.txt")
    data_tsmom_encoder_decoder = (X_enc_arr_train, X_dec_arr_train, y_dec_arr_train, X_enc_arr_valid, X_dec_arr_valid, y_dec_arr_valid, X_enc_arr_test, X_dec_arr_test, y_dec_arr_test, X_enc_dec_indices_arr_train, y_dec_indices_arr_train, X_enc_dec_indices_arr_valid, y_dec_indices_arr_valid, X_enc_dec_indices_arr_test, y_dec_indices_arr_test)
    data_tsmom_decoder = (X_arr_train, y_arr_train, X_arr_valid, y_arr_valid, X_arr_test, y_arr_test, X_arr_indices_train, y_arr_indices_train, X_arr_indices_valid, y_arr_indices_valid, X_arr_indices_test, y_arr_indices_test)
    return (data_tsmom_encoder_decoder, data_tsmom_decoder)

def descriptive_statistics(df, save=True):
    stats = pd.DataFrame(columns = ["Asset", "Min Year", "Asset Class", "Geography"], index = df.index)
    stats["Asset"] = df["ASSET"]
    stats["Min Year"] = df["START_DATE"]
    stats["Asset Class"] = df["ASSET_CLASS"]
    stats["Geography"] = df["GEOGRAPHY"]
    if save:
        table_latex = stats.to_latex(index=True, float_format="{:.2f}".format, bold_rows=True)
        with open("./figures_tables/Tables/Multi-Asset Data Set - Descriptive Statistics.txt", 'w') as file:
            file.write(table_latex)
    print("\nDescriptive Statistics:\n", stats)
    return stats

def summary_statistics(df, save=True):
    stats = pd.DataFrame(columns = ["Start Date", "End Date", "Obs", "Unique", "Mean", "Median", "Std", "Min", "Max", "Q1", "Q3"], index = df.columns)
    for col in df.columns:
        stats.loc[col, "Start Date"] = df.index[0].strftime('%Y-%m-%d')
        stats.loc[col, "End Date"] = df.index[-1].strftime('%Y-%m-%d')
        stats.loc[col, "Obs"] = len(df[col])
        stats.loc[col, "Unique"] = len(pd.unique(df[col]))
        stats.loc[col, "Mean"] = round(df[col].mean(),2)
        stats.loc[col, "Median"] = round(df[col].median(),4)
        stats.loc[col, "Std"] = round(df[col].std(),4)
        stats.loc[col, "Min"] = round(df[col].min(),4)
        stats.loc[col, "Max"] = round(df[col].max(),4)
        stats.loc[col, "Q1"] = round(df[col].quantile(0.25),4)
        stats.loc[col, "Q3"] = round(df[col].quantile(0.75),4)
    if save:
        table_latex = stats.to_latex(index=True, float_format="{:.4f}".format, bold_rows=True)
        with open("./figures_tables/Tables/Multi-Asset Data Set - Descriptive Statistics.txt", 'w') as file:
            file.write(table_latex)
    assert 1!=1
    print("\nSummary Statistics:\n", stats)
    return stats

def plot_cumulative_returns(df, save=True):
    cumulative_returns = (df+1).cumprod()-1
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(cumulative_returns) # label=df.columns
    ax.set_xlabel("Time", size=12)
    ax.set_ylabel("Cumulative Return", size=12)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([-1,0,1,3,5,10,15,20])
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper right', ncol=6, fontsize='small', frameon=False)
    plt.margins(x=0)
    if save:
        plt.savefig("./figures_tables/Multi-Asset Data Set - Cumulative Returns.pdf")

def plot_model_learning(model_fits, model_labels, strategy, model_name, save=True):
    for model_fit, model_label in zip(model_fits, model_labels):
        plt.plot(model_fit.history['loss'], label=model_label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(f"figures_tables/{strategy} - {model_name} - Model Backtesting - Model Learning.pdf") if save else plt.show()
    plt.close()

def model_evaluation(model, daily_returns, model_weights, rebalance_freq, rebalance_days, rebalance_days_evaluation, vol_lookback, vol_target):
    # Model Parameters
    num_assets = np.count_nonzero(model_weights[0])
    # Daily Model Weights
    model_weights_daily = pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns)
    model_weights_daily = model_weights_daily.loc[rebalance_days[0]:rebalance_days_evaluation[-1]]
    model_weights_daily.loc[rebalance_days, :] = model_weights
    model_weights_daily.fillna(method='ffill', inplace=True)
    # Daily (Annualized) Volatilities
    daily_volatility = daily_returns.ewm(span=vol_lookback, min_periods=vol_lookback).std().fillna(method="bfill")
    annualised_volatility = daily_volatility.shift(1) * np.sqrt(252)
    # volatility = annualised_volatility[annualised_volatility.index.isin(rebalance_days_evaluation)]
    volatility = annualised_volatility.loc[rebalance_days[0]:rebalance_days_evaluation[-1]]
    # if rebalance_freq > 1:
    #     returns = daily_returns.rolling(rebalance_freq).apply(lambda x: (1+x).prod()-1) 
    # else:
    returns = daily_returns.copy()
    # Filter for Evaluation Days
    # returns = returns[returns.index.isin(rebalance_days_evaluation)]
    returns = returns.loc[rebalance_days[0]:rebalance_days_evaluation[-1]]
    # Scale Returns to Target Volatility
    returns_volatility_scaled = (vol_target / volatility.values) * returns.values
    # Transform to DataFrames
    # returns_volatility_scaled = pd.DataFrame(index=rebalance_days_evaluation, columns=daily_returns.columns, data=returns_volatility_scaled)
    # weights = pd.DataFrame(index=rebalance_days_evaluation, columns=daily_returns.columns, data=model_weights)
    weights = model_weights_daily.copy()
    returns_volatility_scaled = pd.DataFrame(index=weights.index, columns=weights.columns, data=returns_volatility_scaled)
    # Calculate Model Returns
    returns_model = (1/num_assets) * (weights*returns_volatility_scaled).sum(axis=1)
    # Volatility Scaling on Portfolio Level
    daily_model_volatility = returns_model.ewm(span=vol_lookback, min_periods=vol_lookback).std().fillna(method="bfill")
    annualised_model_volatility = daily_model_volatility.shift(1) * np.sqrt(252)
    returns_model_volatility_scaled = (vol_target / annualised_model_volatility.values) * returns_model.values
    returns_model_volatility_scaled = returns_model_volatility_scaled[1:]
    # returns_model_volatility_scaled = pd.Series(index=rebalance_days_evaluation, data=returns_model_volatility_scaled.reshape(returns_model_volatility_scaled.shape[0]))
    returns_model_volatility_scaled = pd.Series(index=returns_volatility_scaled.index[1:], data=returns_model_volatility_scaled)
    # Cumulative Model Evaluation Value & Scaled Return
    scaled_cumulative_return = (1+returns_model_volatility_scaled).cumprod()
    scaled_cumulative_return[0] = 1
    print(f"\nScaled Cumulative Return ({model}):\n", scaled_cumulative_return)
    return returns_model_volatility_scaled, scaled_cumulative_return

def model_performance_statistics(model_returns, labels, strategy, save=True):
    # Initialize Summary DataFrame
    performance_statistics = pd.DataFrame(
        index=['Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Downside Risk', 'Max Drawdown', 'Calmar Ratio', '% Positive Returns', 'Profit-Loss Ratio'], columns=labels)
    for i, model_return in enumerate(model_returns):
        # Return
        performance_statistics.loc['Ann. Return', labels[i]] = annual_return(model_return)
        # Volatility
        performance_statistics.loc['Ann. Volatility', labels[i]] = annual_volatility(model_return)
        # Sharpe Ratio
        performance_statistics.loc['Sharpe Ratio', labels[i]] = sharpe_ratio(model_return)
        # Downside Deviation
        performance_statistics.loc['Downside Risk', labels[i]] = downside_risk(model_return)
        # Max Drawdown
        performance_statistics.loc['Max Drawdown', labels[i]] = -max_drawdown(model_return)
        # Sortino Ratio
        performance_statistics.loc['Sortino Ratio', labels[i]] = sortino_ratio(model_return)
        # Calmar Ratio
        performance_statistics.loc['Calmar Ratio', labels[i]] = calmar_ratio(model_return)
        # % Positive Returns
        performance_statistics.loc['% Positive Returns', labels[i]] = len(model_return[model_return > 0.0]) / len(model_return)
        # Porfit Loss Ratio
        performance_statistics.loc['Profit-Loss Ratio', labels[i]] = np.mean(model_return[model_return > 0.0]) / np.mean(np.abs(model_return[model_return < 0.0]))
    if save:
        table_latex = performance_statistics.to_latex(index=True, float_format="{:.4f}".format, bold_rows=True)
        with open(f"./figures_tables/{strategy} - Model Backtesting - Performance Evaluation.txt", 'w') as file:
            file.write(table_latex)
    # Print Performance Statistics
    print("Model Evaluation:\n", performance_statistics)
    return performance_statistics

def plot_model_performances(scaled_cumulative_returns, labels, strategy, save=False):
    fig, ax = plt.subplots(figsize = (10,5))
    for i, scaled_cumulative_return in enumerate(scaled_cumulative_returns):
        ax.plot(scaled_cumulative_return.index, scaled_cumulative_return.values, label=labels[i])
    ax.set_xlabel("Time", size=12)
    ax.set_ylabel("Return", size=12)
    ax.set_title("Cumulative Return (OOS)", size = 16)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.legend(loc = "upper left", frameon = False)
    plt.tight_layout()
    plt.savefig(f"figures_tables/{strategy} - Model Backtesting - Cumulative Returns.pdf") if save else plt.show()
    plt.close()
    
def plot_model_performances_log_scale(scaled_cumulative_returns, labels, strategy, save=False):
    fig, ax = plt.subplots(figsize = (10,5))
    for i, scaled_cumulative_return in enumerate(scaled_cumulative_returns):
        ax.plot(scaled_cumulative_return.index, scaled_cumulative_return.values, label=labels[i])
    ax.set_xlabel("Time", size=12)
    ax.set_ylabel("Return", size=12)
    ax.set_title("Cumulative Return (OOS)", size = 16)
    ax.set_yscale('log')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.legend(loc = "upper left", frameon = False)
    plt.tight_layout()
    plt.savefig(f"figures_tables/{strategy} - Model Backtesting - Cumulative Returns (Log-Scale).pdf") if save else plt.show()
    plt.close()

def plot_asset_class_weights(model_weights, df_assets, rebalance_days_evaluation, strategy, save=True):
    # Model Parameters
    num_assets = np.count_nonzero(model_weights[0])
    # Initialize Weights
    weights_df = pd.DataFrame(index=rebalance_days_evaluation, columns=df_assets.index, data=model_weights/num_assets)
    # Initialize Asset Class Weights Dict
    weights_asset_classes = {
        "EQUITIES (EQ)": 0,
        "FIXED INCOME (FI)": 0,
        "FOREIGN EXCHANGE (FX)": 0,
        "COMMODITIES (CM)": 0}
    # Model Parameters
    num_assets = np.count_nonzero(model_weights[0])
    for day in weights_df.index:
        for asset in weights_df.index:
            asset_class = df_assets.loc[asset, "Asset Class"]
            weights_asset_classes.loc[asset_class] += weights_df.loc[day, asset]
    df_weights_asset_classes = pd.DataFrame(weights_asset_classes)
    fig, ax = plt.subplots(figsize = (10,5))
    ax.stackplot(rebalance_days_evaluation, df_weights_asset_classes.T, labels=df_weights_asset_classes.columns)
    ax.set_xlabel("Time", size=12)
    ax.set_ylabel("Weight", size=12)
    ax.set_title("Asset Class Weights", size = 16)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.legend(loc = "upper left", frameon = False)
    plt.tight_layout()
    plt.savefig(f"figures_tables/Asset Class Weights ({strategy}).pdf") if save else plt.show()
    plt.close()

def create_labels(returns):
    modified_returns = returns.copy()
    for day_returns in modified_returns:
        largest_indices = np.argpartition(day_returns, -10)[-10:]
        smallest_indices = np.argpartition(day_returns, 10)[:10]
        day_returns[largest_indices] = 1
        day_returns[smallest_indices] = -1
        day_returns[np.setdiff1d(np.arange(day_returns.size), np.concatenate((largest_indices, smallest_indices)))] = 0
    return modified_returns

def modify_weights(weights, n=10):
    modified_weights = weights.copy()
    for day_weights in modified_weights:
        largest_indices = np.argpartition(day_weights, -n)[-n:]
        smallest_indices = np.argpartition(day_weights, n)[:n]
        day_weights[largest_indices] = 1/(2*n)
        day_weights[smallest_indices] = -1/(2*n)
        day_weights[np.setdiff1d(np.arange(day_weights.size), np.concatenate((largest_indices, smallest_indices)))] = 0
    return modified_weights

def csmom_weights(weights, portfolio, n):
    modified_weights = weights.copy()
    for day_weights in modified_weights:
        largest_indices = np.argpartition(day_weights, -n)[-n:]
        smallest_indices = np.argpartition(day_weights, n)[:n]
        day_weights[largest_indices] = 1/(2*n) if portfolio == "long" else 0
        day_weights[smallest_indices] = -1/(2*n) if portfolio == "short" else 0
        day_weights[np.setdiff1d(np.arange(day_weights.size), np.concatenate((largest_indices, smallest_indices)))] = 0
    return modified_weights