import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal


class LPV_DS_Model:
    def __init__(self, mat_path):
        self.mat_data = loadmat(r'./test_gmm.mat')
        self._load_gmm()
        self._load_lpv_ds()
        self.x_test = self.mat_data['x_test']  # shape (d, M)
        self.x_dot_matlab = self.mat_data['x_dot'].flatten()
        self.pk_x_matlab = self.mat_data['Pk_x'].flatten()

    def _load_gmm(self):
        gmm_raw = self.mat_data['gmm']
        self.gmm = {
            'Priors': gmm_raw['Priors'][0, 0].flatten(),
            'Mu': gmm_raw['Mu'][0, 0],
            'Sigma': gmm_raw['Sigma'][0, 0]
        }

    def _load_lpv_ds(self):
        self.A_g = self.mat_data['A_g']
        self.b_g = self.mat_data['b_g']

    def compute_responsibilities(self, x=None, normalized=True):
        if x is None:
            x = self.x_test
        d, M = x.shape
        K = self.gmm['Priors'].shape[0]
        Px_k = np.zeros((K, M))
        for k in range(K):
            mvn = multivariate_normal(mean=self.gmm['Mu'][:, k], cov=self.gmm['Sigma'][:, :, k])
            Px_k[k, :] = mvn.pdf(x.T) + np.finfo(float).eps
        alpha_Px_k = self.gmm['Priors'][:, np.newaxis] * Px_k
        if normalized:
            return alpha_Px_k / np.sum(alpha_Px_k, axis=0, keepdims=True)
        return alpha_Px_k

    def evaluate_lpv_ds(self, x=None):
        if x is None:
            x = self.x_test
        d, M = x.shape
        K = self.gmm['Priors'].shape[0]
        beta_k_x = self.compute_responsibilities(x)
        x_dot = np.zeros((d, M))
        for i in range(M):
            if self.b_g.shape[1] > 1:
                f_g = np.zeros((d, K))
                for k in range(K):
                    f_g[:, k] = beta_k_x[k, i] * (self.A_g[:, :, k] @ x[:, i] + self.b_g[:, k])
                x_dot[:, i] = np.sum(f_g, axis=1)
            else:
                x_dot[:, i] = self.A_g @ x[:, i] + self.b_g.flatten()
        return x_dot

    def compare_responsibilities(self):
        pk_x_python = self.compute_responsibilities().flatten()
        print("\nResponsibility Comparison:")
        print("MATLAB gamma_k(x):", self.pk_x_matlab)
        print("Python gamma_k(x):", pk_x_python)
        print("Difference:", np.abs(self.pk_x_matlab - pk_x_python))

    def compare_lpv_ds(self):
        x_dot_python = self.evaluate_lpv_ds().flatten()
        print("\nLPV-DS Output Comparison:")
        print("MATLAB x_dot:", self.x_dot_matlab)
        print("Python x_dot:", x_dot_python)
        print("Difference:", np.abs(self.x_dot_matlab - x_dot_python))


# === Example usage ===
if __name__ == "__main__":
    model = LPV_DS_Model('./test_gmm.mat')
    model.compare_responsibilities()
    model.compare_lpv_ds()
    

