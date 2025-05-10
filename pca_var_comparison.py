import numpy as np
import time

def gen_corr_matrix(n=10):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 0.9 ** abs(i - j)
    matrix += np.eye(matrix.shape[0]) * 1e-6 #positive definiteness through regularization
    return matrix

def ensure_positive_definite(corr):
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6) #make any small or neg eigenval as a positive
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def reduce_corr_pca(corr, k=3):
    eigvals, eigvecs = np.linalg.eigh(corr)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    top_eigenvectors = eigvecs[:, :k]
    pca_corr = np.dot(np.dot(top_eigenvectors, np.diag(eigvals[:k])), top_eigenvectors.T)
    return pca_corr

def basket_simulation(n_simulations=50000, corr=None):
    n = corr.shape[0]
    s0 = 100
    sigma = 0.3
    T = 1.0
    dt = 1/252
    steps = int(T / dt)

    L = np.linalg.cholesky(corr)
    results = []

    for _ in range(n_simulations):
        Z = np.random.randn(steps, n)
        dW = np.dot(Z, L.T) * np.sqrt(dt)
        W = np.cumsum(dW, axis=0)
        S = s0 * np.exp(sigma * W)
        avg_price = np.mean(S[-1])
        payoff = max(avg_price - 100, 0)
        results.append(payoff)

    results = np.array(results)
    premium = np.mean(results)
    losses = results - premium
    VaR_95 = np.percentile(losses, 95)
    return premium, VaR_95

corr_original = gen_corr_matrix()
corr_original = ensure_positive_definite(corr_original)

start_orig = time.time()
premium_orig, VaR_orig = basket_simulation(n_simulations=50000, corr=corr_original)
elapsed_orig = time.time() - start_orig

reduced_corr = reduce_corr_pca(corr_original, k=3)
reduced_corr = ensure_positive_definite(reduced_corr)

start_pca = time.time()
premium_pca, VaR_pca = basket_simulation(n_simulations=50000, corr=reduced_corr)
elapsed_pca = time.time() - start_pca

print("Results after 50,000 Simulations\n")

print("Original Correlation Matrix")
print(f"Basket price: {premium_orig:.4f}")
print(f" initial 95% VaR: {VaR_orig:.4f}")
print(f" Compute time: {elapsed_orig:.2f} secs\n")

print("Reduced 3x3 Correlation Matrix (PCA)")
print(f"New Basket price: {premium_pca:.4f}")
print(f"new 95% Var: {VaR_pca:.4f}")
print(f" Compute time: {elapsed_pca:.2f} secs\n")
