global_params = {
    "start_date": "1999-01-01",
    "end_date": "2022-12-31",
    "train_test_split": 0.8,
    "train_validation_split": 0.9,
    "vol_lookback": 63,
    "vol_target": 0.15,
    "days_per_month":  21,
    "months_per_year": 12}

asset_class_map = {
    "EQUITIES (EQ)": 0, 
    "FIXED INCOME (FI)": 1, 
    "FOREIGN EXCHANGE (FX)": 2, 
    "COMMODITIES (CM)": 3}