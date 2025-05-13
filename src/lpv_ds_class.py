import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal


class LPV_DS_Model:
    def __init__(self, mat_path):
        self.mat_data = loadmat(mat_path)
        # what is inside the mat file
        print(self.mat_data.keys())

        self._load_gmm()
        self._load_lpv_ds()

        self.Xi_dot_ref = self.mat_data['Xi_dot_ref']
        self.Xi_ref = self.mat_data['Xi_ref']

        if 'x_test' in self.mat_data:

            self.x_test = self.mat_data['x_test']  # shape (d, M)
            self.x_dot_matlab = self.mat_data['x_dot'].flatten()
            self.pk_x_matlab = self.mat_data['Pk_x'].flatten()

    def _load_gmm(self):
        gmm_raw = self.mat_data['ds_gmm']
        self.gmm = {
            'Priors': gmm_raw['Priors'][0, 0].flatten(),
            'Mu': gmm_raw['Mu'][0, 0],
            'Sigma': gmm_raw['Sigma'][0, 0]
        }

    def _load_lpv_ds(self):
        self.A_g = self.mat_data['A_k']
        self.b_g = self.mat_data['b_k']

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

        x = [[1.5], [3. ]]
        x = np.array(x)
        x_dot_python = self.evaluate_lpv_ds().flatten()
        print("\nLPV-DS Output Comparison:")
        print("MATLAB x_dot:", self.x_dot_matlab)
        print("Python x_dot:", x_dot_python)
        print("Difference:", np.abs(self.x_dot_matlab - x_dot_python))

    
    def visualize_lpv_ds(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        
        # sample some data points x. and compute x_dot

        range_x = np.linspace(-5, 5, 100)
        range_y = np.linspace(-20, 20, 100)

        for i in range(len(range_x)):
            for j in range(len(range_y)):
                x = np.array([[range_x[i]], [range_y[j]]])
                x_dot = self.evaluate_lpv_ds(x)
                plt.quiver(x[0], x[1], x_dot[0], x_dot[1], angles='xy', scale_units='xy', scale=1, color='r')
        plt.xlim(-5, 5)
        plt.ylim(-20, 20)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('LPV-DS Vector Field')
        plt.grid()
        plt.show()


    def visualize_lpv_streamplot(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # Create a grid of points
        x_vals = np.linspace(-2, 2, 20)
        y_vals = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x_vals, y_vals)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Evaluate the vector field at each grid point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([[X[i, j]], [Y[i, j]]])
                x_dot = self.evaluate_lpv_ds(x)
                U[i, j] = x_dot[0, 0]
                V[i, j] = x_dot[1, 0]

        # Plot the streamlines
        plt.figure(figsize=(6, 6))
        plt.streamplot(X, Y, U, V, color='black', linewidth=1)

        # Axis labels with LaTeX
        plt.xlabel(r'$x_1$', fontsize=18)
        plt.ylabel(r'$x_2$', fontsize=18)


        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(False)
        plt.show()


    def visualize_vector_field_with_trajectories(self, initial_conditions, goal):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.integrate import solve_ivp

        # Grid for streamplot
        x_vals = np.linspace(-5, 5, 20)
        y_vals = np.linspace(-5, 10, 40)
        X, Y = np.meshgrid(x_vals, y_vals)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([[X[i, j]], [Y[i, j]]])
                x_dot = self.evaluate_lpv_ds(x)
                U[i, j] = x_dot[0, 0]
                V[i, j] = x_dot[1, 0]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.streamplot(X, Y, U, V, color='gray', linewidth=1, arrowsize=1)

        # Plot trajectories from initial conditions
        for x0 in initial_conditions:
            def dyn(t, x):
                x_vec = np.array([[x[0]], [x[1]]])
                dx = self.evaluate_lpv_ds(x_vec)
                return dx.flatten()

            sol = solve_ivp(dyn, [0, 10], x0, t_eval=np.linspace(0, 5, 300))

            ax.plot(sol.y[0], sol.y[1], 'r-', linewidth=1.5)  # main trajectory
            ax.plot(sol.y[0], sol.y[1], 'k--', linewidth=1)   # optional: overplot for dual color

            # Start point
            ax.plot(x0[0], x0[1], 'go', markersize=6)

        #plot xi_ref
        print("xi_ref", self.Xi_dot_ref.shape)

        for i in range(self.Xi_ref.shape[1]):
            xi_ref = self.Xi_ref[:, i]
            print("xi_ref", xi_ref)
            ax.plot(xi_ref[0], xi_ref[1], 'bo', markersize=2)


        # Plot goal point
        ax.plot(goal[0], goal[1], 'yo', markersize=12)

        ax.set_title("Vector Field and Trajectory", fontsize=14)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.axis('equal')
        plt.show()




# === Example usage ===
if __name__ == "__main__":
    model = LPV_DS_Model('./gmm.mat')
    # model.compare_responsibilities()
    # model.compare_lpv_ds()

    model.visualize_lpv_streamplot()

    initial_conditions = [[-1, -1], [0, 0], [1, 1]]
    goal = [2, 2]
    model.visualize_vector_field_with_trajectories(initial_conditions, goal)
    

