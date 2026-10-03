import math
from datetime import datetime
import numpy as np
import pandas as pd

##############################################################################################
# SORTING DATA FILES:
###############################################################################################

# 1. Data Collecting (Data Frames):
surface_data = pd.read_excel("SPX Implied Volatility Surface Data.xlsx")
spx_price_data = pd.read_excel("Actual Project Data.xlsx", sheet_name="SPX Index").iloc[:, 0:2]
vix_data = pd.read_excel("Actual Project Data.xlsx", sheet_name="VIX Index").iloc[:, 0:2]

# 2. Converting To Matrices and Vectors:
spx_price_mat = spx_price_data.to_numpy()
vix_mat = vix_data.to_numpy()
surface_matrix = surface_data.to_numpy()
spx_price_vector = spx_price_data.iloc[:, 1].to_numpy()
vix_vector = vix_data.iloc[:, 1].to_numpy()

# 3. Convert Date Columns to Datetime Objects:
surface_matrix[:, 0] = pd.to_datetime(surface_matrix[:, 0], format="%d-%b-%Y", errors="coerce")
spx_price_mat[:, 0] = pd.to_datetime(spx_price_mat[:, 0], format="%Y/%m/%d", errors="coerce")
vix_mat[:, 0] = pd.to_datetime(vix_mat[:, 0], format="%Y/%m/%d", errors="coerce")

########################################################################################################
#  DATA CLEANING FOR VolGAN: 
########################################################################################################

# 1. Using Data For Specific Time Period:
price_dates = spx_price_mat[1:, 0].astype('datetime64[D]')
days_2010 = (price_dates >= np.datetime64('2009-12-02')) & (price_dates <= np.datetime64('2010-12-31'))
prices_mat_2010 = spx_price_mat[1:][days_2010, :]

# 2. Full Series Log Returns:
underlying_return = np.log(prices_mat_2010[1:, 1].astype(float) / prices_mat_2010[:-1, 1].astype(float))

# 3. Calculating the 21-Day Rolling Volatility:
gamma_2010 = np.zeros(prices_mat_2010.shape[0] - 21)
k = 21
for i in range(len(gamma_2010)):
    
    gamma_2010[i] = np.sqrt((252/21) * sum(np.square(underlying_return[i:k])))
    k = k + 1

# 4. Log Surface Matrix:
scaled_surface_matrix = surface_matrix[:, 1:].astype(float) * 0.01
log_surface_matrix = np.log(scaled_surface_matrix)

##########################################################################################################
# ARBITRAGE OPPORTUNITY CALCULATION:
##########################################################################################################


##########################################################################################################
# Creating The VolGAN Model:
##########################################################################################################

# def Generator():
























