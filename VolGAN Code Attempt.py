import os
from typing import Any
from datetime import datetime

# Core Math & Computing
import numpy as np
import numpy.random as rnd
import pandas as pd
from tqdm import tqdm

# Deep Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim

# Market Data & Analysis
import yfinance as yf
import pandas_datareader as pd_data
from pandas_datareader import data as pdr
from statsmodels.tsa.stattools import acf, pacf

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def penalty_arbitrage(C, tau_vec, m_vec):

    d_tau = tau_vec[1:] - tau_vec[:-1]
    d_m = m_vec[1:] - m_vec[:-1]
    tau_j = tau_vec[:-1]
    d_m_mid = (m_vec[2:] - m_vec[:-2]) / 2.0

    if C.dim() == 3:
        batch_d_tau = d_tau.view(1, 1, -1)
        batch_d_m = d_m.view(1, -1, 1)
        batch_tau_j = tau_j.view(1, 1, -1)
        d_m_mid_b = d_m_mid.view(1, -1, 1)
    else:
        batch_d_tau = d_tau.view(1, -1)
        batch_d_m = d_m.view(-1, 1)
        batch_tau_j = tau_j.view(1, -1)
        d_m_mid_b = d_m_mid.view(-1, 1)

    # 1. Calendar Violation (P1): Increasing in maturity tau
    dC_dtau = ((C[..., 1:] - C[..., :-1]) / batch_d_tau) * batch_tau_j
    p_1 = torch.relu(-dC_dtau).sum()

    # 2. Call Price / Monotonicity Violation (P2): Decreasing in moneyness m
    dC_dm = (C[..., 1:, :] - C[..., :-1, :]) / batch_d_m
    p_2 = torch.relu(dC_dm).sum()

    # 3. Butterfly Violation (P3): Convex in moneyness m
    d2C_dm2 = (
        (C[..., 2:, :] - C[..., 1:-1, :]) / batch_d_m[..., 1:, :]
        - (C[..., 1:-1, :] - C[..., :-1, :]) / batch_d_m[..., :-1, :]
    ) / d_m_mid_b
    p_3 = torch.relu(-d2C_dm2).sum()

    total_penalty = p_1 + p_2 + p_3

    return p_1, p_2, p_3, total_penalty

def Gaussian_CDF(x):
    return 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))

def Black76_OptionPrice(St, tau, F, K, sigma):
    efact = F/St
    d1 = (-np.log(K/St) + tau*0.5*sigma**2)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    price = efact*(F*Gaussian_CDF(d1) - K*Gaussian_CDF(d2))
    return price

def BS_optionPrice(St, tau, r, K, sigma):
    d1 = (-np.log(K / St) + tau * (r + 0.5 * sigma ** 2)) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = St*Gaussian_CDF(d1) - K*Gaussian_CDF(d2)*np.exp(-r*tau)
    return price

def BS_OptionPrice_tensor(St,tau,r,K,sigma):

    norm = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    d1 = (torch.log(St/K)+tau*(r+0.5*sigma*sigma))/(sigma*torch.sqrt(tau))
    d2 = d1-sigma*torch.sqrt(tau)
    price = St*norm.cdf(d1)-K*norm.cdf(d2)*torch.exp(-r*tau)
    price[price<=0] = 10**(-10)

    return price


import torch


def smallBS_unified(m, tau, sigma, r):

    norm = torch.distributions.Normal(0.0, 1.0)

    sqrt_tau = torch.sqrt(tau)
    d1 = (-torch.log(m) + tau * (r + 0.5 * sigma ** 2)) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    price = norm.cdf(d1) - m * norm.cdf(d2) * torch.exp(-r * tau)

    return torch.clamp(price, min=1e-10)

def RelativeCall_tensor(m,tau,sigma):
    norm = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    d1 = (-torch.log(m)+tau*0.5*sigma*sigma)/(sigma*torch.sqrt(tau))
    d2 = d1-sigma*torch.sqrt(tau)
    price = norm.cdf(d1)-m*norm.cdf(d2)
    price[price<=0] = 10**(-10)
    return price

# Functions For Converting Between K, St and tau, m:
def K_T_to_mu_tau(K,T,St,t):
    return K/St,T-t
def mu_tau_to_K_T(mu,tau,St,t):
    return mu*St,tau+t

# Interpolating Points Using Strike K:
def get_points(I_known, spatial_known, tau_known, spatial_want, tau_want, use_moneyness=False):

    interpolator = RegularGridInterpolator(
        (spatial_known, tau_known),
        I_known,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    grid_spatial, grid_tau = np.meshgrid(spatial_want, tau_want, indexing='ij')
    return interpolator((grid_spatial, grid_tau))

# Function To Plot The Surface in 3-D:
def plot_surface_interactive(X, Y, Z, xlabel, ylabel, zlabel, title):

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
        ),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()

# Plotting the Scatter Points in 3-D:
def scatter_surface_interactive(X, Y, Z, xlabel, ylabel, zlabel, title="3D Scatter Plot"):

    fig = go.Figure(data=[
        go.Scatter3d(
            x=X.flatten() if hasattr(X, 'flatten') else X,
            y=Y.flatten() if hasattr(Y, 'flatten') else Y,
            z=Z.flatten() if hasattr(Z, 'flatten') else Z,
            mode='markers',
            marker=dict(
                size=4,
                color=Z.flatten() if hasattr(Z, 'flatten') else Z,  # Colors points by height/volatility
                colorscale='Viridis',
                opacity=0.8,
                showscale=True  # Displays the colorbar
            )
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
        ),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()

# Functions For Converting Between 1-D and 2-D:
def entangle_kt_fast(surface):
    return surface.flatten(order='F')  # or surface.T.reshape(-1)

def detangle_kt_fast(surface_vector, lk, lt):
    return surface_vector.reshape((lk, lt), order='F')

def detangle_kt_torch_fast(surface_tensor, lk, lt):
    return surface_tensor.view(lt, lk).T

# Creating the Generator:

class Generator(nn.Module):

    def __init__(self, noise_dim, cond_dim, hidden_dim, output_dim, mean_in = None, std_in = None, mean_out = None, std_out = None):
        super(Generator, self).__init__()
        self.input_dim = noise_dim + cond_dim
        self.register_buffer('mu_i', mean_in if mean_in is not None else torch.tensor(0.0))
        self.register_buffer('std_i', std_in if std_in is not None else torch.tensor(1.0))
        self.register_buffer('mu_o', mean_out if mean_out is not None else torch.tensor(0.0))
        self.register_buffer('std_o', std_out if std_out is not None else torch.tensor(1.0))

        self.net = nn.Sequential(

            nn.Linear(self.input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Softplus(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid()

        )

        def forward(self, noise, condition):

            norm_condition = (condition - self.mu_i) / (self.std_i + 1e-8)
            x = torch.cat([noise, norm_condition], dim=-1)
            out = self.net(x)
            out = self.mu_o + (self.std_o * out)

            return out


# Creating the Discriminator NN:
class Discriminator(nn.Module):

    def __init__(self, in_dim, hidden_dim, mean=None, std=None):
        super().__init__()

        self.input_dim = in_dim
        self.hidden_dim = hidden_dim

        self.register_buffer('mu_i', mean if mean is not None else torch.tensor(0.0))
        self.register_buffer('std_i', std if std is not None else torch.tensor(1.0))

        self.net = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
            nn.Softplus(),
            nn.Linear(in_features=self.hidden_dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, in_chan):

        x = (in_chan - self.mu_i) / (self.std_i + 1e-8)
        out = self.net(x)

        return out
























































