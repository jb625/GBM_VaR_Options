import numpy as np

n = 10
s0 = 100
sigma = 0.3
T = n/252 #n over num trading days
dt = T/(1*n) 
steps = int(T / dt)
simulations = 10000

corr = np.fromfunction(lambda i, j: 0.9 ** np.abs(i - j), (n, n))
L = np.linalg.cholesky(corr)

payoffs = []

for i in range(simulations):
    z = np.random.randn(steps, n)
    dz = z @ L.T * np.sqrt(dt)
    W = np.cumsum(dz, axis=0)
    t = np.linspace(dt, T, steps)[:, None]  # shape (steps, 1)
    S = s0 * np.exp(-0.5 * sigma**2 * t + sigma * W)
    avg = np.mean(S[-1]) #last point in the 2d array
    payoffs.append(max(avg - 100, 0)) #atm strike price

payoffs = np.array(payoffs)
premium = np.mean(payoffs) #same price at t=0 and t=T, because no discounting
print(f"Basket call price: ", premium)


pnl = premium - payoffs
#print(pnl)
var_95 = (np.percentile(pnl, 5)) 

print("VaR:" f"{var_95:.4f}")
