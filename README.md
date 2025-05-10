# GBM_VaR_Options
Geometric Brownian Motion Option pricing, var95, and PCA

a.There are ten non-Dividend paying stocks following GBM with no drift term (zero interest rate) and 30% annualized volatility and each of them is worth $100 currently.

b.Correlations between each pair of stocks are defined as (0.9)^abs(i – j), where i and j are the stock numbering.

c.We are interested in the price of at the money European basket call option whose payoff is max(avg(S1,…S10)-100,0) and maturity date is in one year.

d.If we sell this option at the market now but receive the premium at the maturity date, what would be the total loss that will not be exceeded at 95% confidence level in one year? This is basically the 95% 1-year VaR.

Extra:
Reduce the size of the above correlation matrix to 3x3 using the principal component analysis or similar method
Repeat the calculation for c) and d) and compare the results and computation time with 50,000 simulation repetitions. 
