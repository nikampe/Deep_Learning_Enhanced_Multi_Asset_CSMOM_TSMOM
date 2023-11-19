strategy_params_csmom = {
    "train_interval": 5,
    "rebalance_freq": 21,
    "num_total_assets": 50,
    "num_assets": 10}

feature_params_csmom = {
    "cumulative_periods": [
        1, 5, 21, 63, 126, 252],
        # 21, 63, 126, 252],
        # int(3*strategy_params_csmom["rebalance_freq"]),
        # int(6*strategy_params_csmom["rebalance_freq"]),
        # int(12*strategy_params_csmom["rebalance_freq"])], # 1,21,63,126,252, # [3*strategy_params_csmom["rebalance_freq"], 6*strategy_params_csmom["rebalance_freq"], 12*strategy_params_csmom["rebalance_freq"]]
    "normalized_periods": [
        1, 5, 21, 63, 126, 252],
        # 21, 63, 126, 252],
        # int(3*strategy_params_csmom["rebalance_freq"]),
        # int(6*strategy_params_csmom["rebalance_freq"]),
        # int(12*strategy_params_csmom["rebalance_freq"])], # [1,21,63,126,252], # [3*strategy_params_csmom["rebalance_freq"], 6*strategy_params_csmom["rebalance_freq"], 12*strategy_params_csmom["rebalance_freq"]]
    "macd_span_short": [8, 16, 32],
    "macd_span_long": [24, 48, 96],
    "macd_std_short": 63,
    "macd_std_long": 252,
    "macd_cumulative_periods": [
        21, 63, 126, 252]}
        # int(3*strategy_params_csmom["rebalance_freq"]),
        # int(6*strategy_params_csmom["rebalance_freq"]),
        # int(12*strategy_params_csmom["rebalance_freq"])]} # [1,3,6,12], # [1*strategy_params_csmom["rebalance_freq"], 3*strategy_params_csmom["rebalance_freq"], 6*strategy_params_csmom["rebalance_freq"], 12*strategy_params_csmom["rebalance_freq"]]

strategy_params_tsmom = {
    "train_interval": 5,
    "rebalance_freq": 21,
    "num_assets": 50}

feature_params_tsmom = {
    "lookback": int(252/strategy_params_tsmom["rebalance_freq"]),
    "cumulative_periods": [
        1, 5, 21, 63, 126, 252],
        # int(3*strategy_params_tsmom["rebalance_freq"]),
        # int(6*strategy_params_tsmom["rebalance_freq"]),
        # int(12*strategy_params_tsmom["rebalance_freq"])], # 1,21,63,126,252
    "normalized_periods": [
        1, 5, 21, 63, 126, 252],
        # int(3*strategy_params_tsmom["rebalance_freq"]),
        # int(6*strategy_params_tsmom["rebalance_freq"]),
        # int(12*strategy_params_tsmom["rebalance_freq"])], # 1,21,63,126,252
    "macd_span_short": [8,16,32],
    "macd_span_long": [24,48,96],
    "macd_std_short": 63,
    "macd_std_long": 252}