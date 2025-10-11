# Functions that compute European option prices using Black-Scholes formula

# input arguments in all cases are:
# S is the spot price 
# K is the strike price 
# T is the time to strike in years
# r is the risk free rate
# sigma is the volatility 

# returned values are price or delta

import numpy as np
from scipy.stats import norm

def BS_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def BS_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -S * norm.cdf(-d1) + K * np.exp(-r*T) * norm.cdf(-d2)

def BS_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    BS_call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    BS_put = BS_call - S + K * np.exp(-r*T)
    return BS_call, BS_put

def BS_call_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    return norm.cdf(d1)

def BS_put_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    return norm.cdf(d1) - 1

def BS_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    return call_delta, put_delta