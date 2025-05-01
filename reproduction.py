import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal

# === Load the .mat file ===
mat = loadmat(r'./test_gmm.mat')  # Adjust path if needed
print("Loaded keys:", list(mat.keys()))

# === Extract GMM ===
gmm_raw = mat['gmm']
gmm = {
    'Priors': gmm_raw['Priors'][0, 0].flatten(),
    'Mu': gmm_raw['Mu'][0, 0],
    'Sigma': gmm_raw['Sigma'][0, 0]
}

# === Extract test point ===
x = mat['x_test']  # (d, 1)

# === Responsibilities from MATLAB ===
Pk_x_matlab = mat['Pk_x'].flatten()

# === Compute responsibilities in Python ===
def compute_responsibilities(x, priors, mu, sigma, normalized=True):
    d, M = x.shape
    K = priors.shape[0]
    Px_k = np.zeros((K, M))
    for k in range(K):
        mvn = multivariate_normal(mean=mu[:, k], cov=sigma[:, :, k])
        Px_k[k, :] = mvn.pdf(x.T) + np.finfo(float).eps
    alpha_Px_k = priors[:, np.newaxis] * Px_k
    return alpha_Px_k / np.sum(alpha_Px_k, axis=0, keepdims=True) if normalized else alpha_Px_k

Pk_x_python = compute_responsibilities(x, gmm['Priors'], gmm['Mu'], gmm['Sigma']).flatten()

print("\nResponsibility Comparison:")
print("MATLAB gamma_k(x):", Pk_x_matlab)
print("Python gamma_k(x):", Pk_x_python)
print("Difference:", np.abs(Pk_x_matlab - Pk_x_python))

# === LPV-DS ===
def lpv_ds(x, gmm, A_g, b_g):
    d, M = x.shape
    K = gmm['Priors'].shape[0]
    beta_k_x = compute_responsibilities(x, gmm['Priors'], gmm['Mu'], gmm['Sigma'])
    x_dot = np.zeros((d, M))
    for i in range(M):
        if b_g.shape[1] > 1:
            f_g = np.zeros((d, K))
            for k in range(K):
                f_g[:, k] = beta_k_x[k, i] * (A_g[:, :, k] @ x[:, i] + b_g[:, k])
            x_dot[:, i] = np.sum(f_g, axis=1)
        else:
            x_dot[:, i] = A_g @ x[:, i] + b_g.flatten()
    return x_dot

# === Extract and evaluate LPV-DS ===
A_g = mat['A_g']
b_g = mat['b_g']
x_dot_matlab = mat['x_dot'].flatten()
x_dot_python = lpv_ds(x, gmm, A_g, b_g).flatten()

print("\nLPV-DS Output Comparison:")
print("MATLAB x_dot:", x_dot_matlab)
print("Python x_dot:", x_dot_python)
print("Difference:", np.abs(x_dot_matlab - x_dot_python))
