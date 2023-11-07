import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
warnings.filterwarnings("ignore")
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Futures
PATH = "data/Futures.xlsx"

def load_assets(start_date):
    assets = pd.read_excel(PATH, sheet_name = "Assets", header = 0, index_col = 0, parse_dates = True)
    assets = assets[assets["START_DATE"] <= start_date]
    return assets

def load_futures(start_date="1999-01-01", end_date="2022-12-31", threshold=0.9, vol_threshold=5):
    path = "/home/niklasleanderkampe/Desktop/Thesis/data/futures/"
    files = [file for file in os.listdir(path) if file.endswith('.CSV')]
    files = sorted(files)
    column_names = []
    futures = pd.DataFrame([])
    assets = pd.DataFrame(columns=["Asset", "Asset Class", "Start Date", "Geography"])
    for file in files:
        future = pd.read_csv(os.path.join(path, file), index_col=0, header=None, parse_dates=True)
        future = future.loc[:,4].replace(0.00, np.nan)
        if future.first_valid_index() <= pd.Timestamp(start_date) and future.last_valid_index() >= pd.Timestamp(end_date):
            futures = pd.concat([futures, future], axis=1)
            identifier = file.replace(".CSV", "")
            column_names.append(identifier)
            assets.loc[identifier, "Asset"] = assets_lookup[identifier][0]
            assets.loc[identifier, "Asset Class"] = assets_lookup[identifier][1]
            assets.loc[identifier, "Start Date"] = datetime.strftime(future.first_valid_index(), "%Y-%m-%d")
            assets.loc[identifier, "Geography"] = assets_lookup[identifier][2]
    # Rename Columns
    futures.columns = column_names
    # Filter on Start and End Date
    futures = futures.iloc[(futures.index>=pd.Timestamp(start_date)) & (futures.index <= pd.Timestamp(end_date))]
    # Sort on the Date Index
    futures.sort_index(axis=0, inplace=True)
    # Drop columns with less than 90% data
    na_proportions = futures.isna().mean()
    drop_assets = na_proportions[na_proportions>(1-threshold)].index
    futures = futures.drop(drop_assets, axis=1)
    assets = assets.drop(drop_assets, axis=0)
    # Drop Columns with Negatiuve Prices
    zero_negative_cols = futures.columns[(futures <= 0).any()]
    futures = futures.drop(zero_negative_cols, axis=1)
    assets = assets.drop(zero_negative_cols, axis=0)
    # Fill NAs with value of last row index
    futures = futures.fillna(method='ffill').dropna(axis=1, how="any")
    # Delete Value larger than 5x EWM Standard Deviation
    for col in futures.columns:
        ewm = futures[col].ewm(halflife=252)
        means = ewm.mean()
        stds = ewm.std()
        futures[col] = np.minimum(futures[col], means + vol_threshold * stds)
        futures[col] = np.maximum(futures[col], means - vol_threshold * stds)
    # Drop NAs
    futures.dropna(axis=0, how="any", inplace=True)
    # Assets DataFrame to Latex Table
    asset_class_order = ['EQUITIES (EQ)', 'FIXED INCOME (FI)', 'FOREIGN EXCHANGE (FX)', 'COMMODITIES (CM)']
    assets['Asset Class'] = pd.Categorical(assets['Asset Class'], categories=asset_class_order, ordered=True)
    assets = assets.sort_values('Asset Class')
    assets.to_latex("figures_tables/Tables/Multi-Asset Data Set - Overview.txt")
    return assets, futures

assets_lookup = {
    "AP": ["Australian Price Index", "FIXED INCOME (FI)", "Australia"],   
    "BC": ["Brent Crude Oil", "COMMODITIES (CM)", "Global"], 
    "FC": ["Feeder Cattle", "COMMODITIES (CM)", "Global"],
    "LC": ["Live Cattle", "COMMODITIES (CM)", "Global"],
    "CC": ["Cocoa", "COMMODITIES (CM)", "Global"],     
    "KC": ["Coffee", "COMMODITIES (CM)", "Global"],       
    "CR": ["CRB Index", "COMMODITIES (CM)", "Global"],
    "HG": ["Copper", "COMMODITIES (CM)", "Global"],    
    "C":  ["Corn", "COMMODITIES (CM)", "Global"],        
    "CT": ["Cotton", "COMMODITIES (CM)", "Global"],       
    "CL": ["Crude Oil", "COMMODITIES (CM)", "Global"],   
    "ED": ["Eurodollar", "FIXED INCOME (FI)", "Euro Zone"],  
    "CB": ["Canada Govt Bond (10yr)", "FIXED INCOME (FI)", "Canada"],
    "GC": ["Gold", "COMMODITIES (CM)", "Global"],         
    "HO": ["Heating Oil", "COMMODITIES (CM)", "Global"],  
    "LH": ["Live Hogs", "COMMODITIES (CM)", "Global"],    
    "BG": ["Brent Gas", "COMMODITIES (CM)", "Global"],   
    "LB": ["Lumber", "COMMODITIES (CM)", "Global"],       
    "EC": ["Eurodollar (Comp)", "FIXED INCOME (FI)", "Euro Zone"],
    "O":  ["Oats", "COMMODITIES (CM)", "Global"],         
    "JO": ["Orange Juice", "COMMODITIES (CM)", "Global"], 
    "PA": ["Palladium", "COMMODITIES (CM)", "Global"],    
    "PL": ["Platinum", "COMMODITIES (CM)", "Global"],     
    "DA": ["Milk", "COMMODITIES (CM)", "Global"],     
    "SP": ["S&P 500, Index", "EQUITIES (EQ)", "United STates of America"],  
    "SI": ["Silver",  "COMMODITIES (CM)", "Global"],       
    "S":  ["Soybeans", "COMMODITIES (CM)", "Global"],     
    "SM": ["Soybeans (Meal)", "COMMODITIES (CM)"],  
    "BO": ["Soybeans (Oil)", "COMMODITIES (CM)", "Global"],  
    "SB": ["Sugar", "COMMODITIES (CM)", "Global"], 
    "SF": ["Swiss Franc", "FOREIGN EXCHANGE (FX)", "Switzerland"],  
    "ER": ["Russell 1000, Mini", "EQUITIES (EQ)", "United States of America"],
    "US": ["US T-Bonds (Comp)", "FIXED INCOME (FI)", "United States of America"],  
    "TY": ["US T-Note (10y) (Comp)", "FIXED INCOME (FI)", "United States of America"], 
    "FB": ["US T-Note (5y) (Comp)", "FIXED INCOME (FI)", "United States of America"], 
    "ZI": ["Silver", "COMMODITIES (CM)", "Global"],      
    "KW": ["Wheat (KC)", "COMMODITIES (CM)", "Global"],    
    "MW": ["Wheat (Mini)", "COMMODITIES (CM)", "Global"],  
    "RB": ["Gasoline", "COMMODITIES (CM)", "Global"],
    "NK": ["Nikkei, Index", "EQUITIES (EQ)", "Japan"],  
    "DX": ["US Dollar Index", "FOREIGN EXCHANGE (FX)", "United States of America"],  
    "NG": ["Natural Gas", "COMMODITIES (CM)", "Global"],  
    "UA": ["US T-Bonds (Day)", "FIXED INCOME (FI)", "United States of America"],   
    "TA": ["US T-Note (10y) (Day)", "FIXED INCOME (FI)", "United States of America"], 
    "AD": ["Australian Dollar", "FOREIGN EXCHANGE (FX)", "Australia"],  
    "MD": ["S&P 400, Index", "EQUITIES (EQ)", "United States of America"],     
    "NR": ["Rough Rice", "COMMODITIES (CM)", "Global"],
    "GI": ["Goldmnan Saks C.I.", "COMMODITIES (CM)", "Global"],  
    "XU": ["Euro Stoxx 50, Index", "EQUITIES (EQ)", "Euro Zone"], 
    "EN": ["Nasdaq, Mini", "EQUITIES (EQ)", "United States of America"], 
    "RL": ["Russell 2000, Index", "EQUITIES (EQ)", "United States of America"],   
    "DJ": ["Dow Jones, Index", "EQUITIES (EQ)", "United States of America"],   
    "ES": ["S&P 500, Mini", "EQUITIES (EQ)", "United States of America"],  
    "ND": ["Nasdaq 100, Index", "EQUITIES (EQ)", "United States of America"],  
    "MP": ["Mexican Peso", "FOREIGN EXCHANGE (FX)", "Mexico"],    
    "SC": ["S&P 500, Comp", "EQUITIES (EQ)", "United States of America"],  
    "AN": ["Australian Dollar (Comp)", "FOREIGN EXCHANGE (FX)", "Australia"],  
    "BN": ["British Pound", "FOREIGN EXCHANGE (FX)", "Great Britian"],  
    "CN": ["Canadian Dollar", "FOREIGN EXCHANGE (FX)", "Canada"],  
    "SN": ["Swiss Franc", "FOREIGN EXCHANGE (FX)", "Switzerland"],  
    "JN": ["Japanese Yen", "FOREIGN EXCHANGE (FX)", "Japan"],    
    "FX": ["Euro", "FOREIGN EXCHANGE (FX)", "Euro Zone"],   
    "FN": ["Euro (Comp)", "FOREIGN EXCHANGE (FX)", "Euro Zone"],  
    "AX": ["DAX 40, Index", "EQUITIES (EQ)", "Germany"],   
    "DT": ["Euro Bond (Bund)", "FIXED INCOME (FI)", "Euro Zone"],  
    "LX": ["FTSE 100, Index", "EQUITIES (EQ)", "Great Britian"],     
    "GS": ["Long Gilt", "FIXED INCOME (FI)", "Great Britian"],    
    "SS": ["Sterling", "COMMODITIES (CM)", "Great Britian"],     
    "FA": ["US T-Note (5y) (Day)", "FIXED INCOME (FI)", "United States of America"],
    "TD": ["US T-Note (2y) (Day)", "FIXED INCOME (FI)", "United States of America"], 
    "TU": ["US T-Note (2y) (Comp)", "FIXED INCOME (FI)", "United States of America"], 
    "YM": ["Mini Dow Jones", "EQUITIES (EQ)", "United States of America"],    
    "ZD": ["Dow Jones, Index", "EQUITIES (EQ)", "United States of America"],
    "XX": ["Dow Jones Stoxx 50", "EQUITIES (EQ)", "United States of America"],  
    "HS": ["Hang Seng, Index", "EQUITIES (EQ)", "Hong Kong"],    
    "CA": ["CAC 40, Index", "EQUITIES (EQ)", "France"], 
    "UB": ["Euro Bobl", "FIXED INCOME (FI)", "Euro Zone"],    
    "UZ": ["Euro Schatz", "FIXED INCOME (FI)", "Euro Zone"],  
    "ZG": ["Gold", "COMMODITIES (CM)", "Global"],    
    "ZC": ["Corn", "COMMODITIES (CM)", "Global"],    
    "ZL": ["Soyoil", "COMMODITIES (CM)", "Global"],  
    "ZM": ["Soymeal", "COMMODITIES (CM)", "Global"], 
    "ZO": ["Oats", "COMMODITIES (CM)", "Global"],    
    "ZR": ["Rough Rice", "COMMODITIES (CM)", "Global"], 
    "ZS": ["Soybeans", "COMMODITIES (CM)", "Global"],
    "ZW": ["Wheat", "COMMODITIES (CM)", "Global"],   
    "ZU": ["Crude Oil", "COMMODITIES (CM)", "Global"],
    "ZB": ["Gasoline", "COMMODITIES (CM)", "Global"],    
    "ZH": ["Heating Oil", "COMMODITIES (CM)", "Global"],
    "ZN": ["Natural Gas", "COMMODITIES (CM)", "Global"],
    "ZK": ["Copper", "COMMODITIES (CM)", "Global"],  
    "ZA": ["Paladium", "COMMODITIES (CM)", "Global"],
    "ZP": ["Platinum", "COMMODITIES (CM)", "Global"],
    "ZF": ["Feeder Cattle", "COMMODITIES (CM)", "Global"], 
    "ZT": ["Live Cattle", "COMMODITIES (CM)", "Global"], 
    "ZZ": ["Lean Hogs", "COMMODITIES (CM)", "Global"]
}

if __name__ == "__main__":
    load_futures()