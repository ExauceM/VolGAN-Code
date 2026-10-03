import math
from datetime import datetime
import numpy as np
import pandas as pd

###############################################################################################
# HOUSEKEEPING: 
###############################################################################################

SURFACE_FILE = "SPX Implied Volatility Surface Data.xlsx"
MARKET_FILE = "Actual Project Data.xlsx"
SPX_SHEET_NAME = "SPX Index"
VIX_SHEET_NAME = "VIX Index"
START_DATE = "2010-01-02"
END_DATE = "2010-12-31"
TENORS = ["1W", "2W", "1M", "3M", "9M", "1Y"]
MONEYNESS = [80, 90, 95, 97.5, 100, 102.5, 105, 110, 120]

###################################################################################################
# DATA CLEANING FUNCTIONS:
###################################################################################################

# 1. Price Function:

def load_price_data(path = MARKET_FILE, Sheet_Name = SPX_SHEET_NAME):
    df1 = pd.read_excel(path, sheet_name = Sheet_Name, usecols = [0, 1])
    df1.columns = ["Date", "Price"]
    df1["Date"] = pd.to_datetime(df1["Date"], errors="coerce")
    df1 = df1.dropna(subset=["Date", "Price"]).drop_duplicates("Date")
    return df1.set_index("Date").sort_index()["Price"].astype(float)

# 2. VIX Function:

def load_vix_data(path = MARKET_FILE, Sheet_Name = VIX_SHEET_NAME):
    df2 = pd.read_excel(path, sheet_name = Sheet_Name, header=None, usecols=[0, 1])
    df2.columns = ["Date", "VIX"]
    df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
    df2 = df2.dropna(subset=["Date", "VIX"]).drop_duplicates("Date")
    return df2.set_index("Date").sort_index()["VIX"].astype(float)

# 3. Surface Function:
def load_surface_data(path = SURFACE_FILE):
    df3 = pd.read_excel(path)
    df3["Date"] = pd.to_datetime(df3["Date"])
    df3 = df3.set_index("Date").sort_index()
    cols = [f"{t}_{m:g}%" for t in TENORS for m in MONEYNESS]
    assert all(c in df3.columns for c in cols)
    sigma = df3[cols] / 100.0                       # percent -> decimal
    assert sigma.notna().all().all() and (sigma > 0).all().all()
    return np.log(sigma), cols 

# 4. Dataset Builder Function:

def build_dataset():
    S = load_price_data()
    g, cols = load_surface_data()
    r = np.log(S / S.shift(1))                                    
    gamma = np.sqrt(252.0 / 21.0 * (r ** 2).rolling(21).sum())    
    cal = S.index                                                 
    prev = pd.Series(cal[:-1], index=cal[1:])                     
    samples = []
    for d in g.index[(g.index >= START_DATE) & (g.index <= END_DATE)]:
        p = prev.get(d)
        if p is None or p not in g.index:                         
            continue
        pp = prev.get(p)
        if pd.isna(gamma.get(p, np.nan)) or pd.isna(r.get(pp, np.nan)):
            continue
        samples.append((d, p, pp))
    dates = pd.DatetimeIndex([s[0] for s in samples])
    p_ = pd.DatetimeIndex([s[1] for s in samples]); pp_ = pd.DatetimeIndex([s[2] for s in samples])
    cond = np.column_stack([r.loc[p_].values,         
                            r.loc[pp_].values,         
                            gamma.loc[p_].values,      
                            g.loc[p_].values])         
    dg = g.loc[dates].values - g.loc[p_].values        
    target = np.column_stack([r.loc[dates].values, dg])
    return dates, cond.astype(np.float32), target.astype(np.float32), cols, S, g, r, gamma 

if __name__ == "__main__":
    import sys
    dates, cond, target, cols, S, g, r, gamma = build_dataset()
    print("samples",len(dates),dates[0].date(),dates[-1].date(),"cond",cond.shape,"target",target.shape)