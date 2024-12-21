import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from kernel import *
from utils import *
import seaborn as sns


class GP:
    def __init__(self, f, kernel, theta, bound, noise=1e-5):
        """

        :param f: y = f(X)
        :param kernel: kernel(X,Y,theta)，其中theta为超参数列表
        :param theta: 超参数初始值
        :param bound: 超参数上下界
        :param noise: 噪声
        """
        self.f = f
        self.kernel = kernel
        self.theta = theta
        self.noise = noise
        self.bound = bound
        self._X = None
        self._y = None
        self._m = None
        self._s = None
        self._X_test = None
        self._y_test = None

    def fit(self, X):
        self._X = X
        self._y = self.f(X)

        def marginal_likelihood(theta):
            K = self.kernel(self._X, self._X, *theta) + self.noise * np.eye(self._X.shape[0])
            return np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
                0.5 * self._y.T @ np.linalg.inv(K) @ self._y + \
                0.5 * self._X.shape[0] * np.log(2 * np.pi)

        res = minimize(marginal_likelihood, x0=self.theta, bounds=self.bound, method='L-BFGS-B')
        self.theta = res.x

    def predict(self, X_test):
        self._X_test = X_test
        C_N = self.kernel(self._X, self._X, *self.theta) + self.noise * np.eye(self._X.shape[0])
        k = self.kernel(self._X, self._X_test, *self.theta)
        c = self.kernel(self._X_test, self._X_test, *self.theta) + self.noise * np.eye(self._X_test.shape[0])
        C_N_inv = np.linalg.inv(C_N)
        m = k.T @ C_N_inv @ self._y
        cov = c - k.T @ C_N_inv @ k
        s = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), 0))
        self._m = m
        self._s = s
        return m, s

    def plot(self):
        plt.plot(self._X_test.flatten(), self._m.flatten(), label=r'$\mu$')
        plt.plot(self._X_test.flatten(), self.f(self._X_test).flatten(), label=r'$f(x)$')
        plt.scatter(self._X.flatten(), self._y.flatten(), label='train data', marker='x', color='black')
        plt.fill_between(self._X_test.flatten(), self._m.flatten() - 1.96 * self._s.flatten(),
                         self._m.flatten() + 1.96 * self._s.flatten(), alpha=0.2, label=r'$\mu\pm 2\sigma$')
        plt.legend()
        plt.show()


class SparseGP:
    def __init__(self, f, kernel, theta, bound, noise=1e-5):
        """

        :param f: y = f(X)
        :param kernel: kernel(X,Y,theta)，其中theta为超参数列表
        :param theta: 超参数初始值
        :param bound: 超参数上下界
        :param noise: 噪声
        """
        self.f = f
        self.kernel = kernel
        self.theta = theta
        self.noise = noise
        self.bound = bound
        self._X = None
        self._y = None
        self._Z = None
        self._m = None
        self._s = None
        self._X_test = None
        self._y_test = None

    def C_N_inv(self, theta):
        K_yy = self.kernel(self._X, self._X, *theta) + self.noise * np.eye(self._X.shape[0])
        K_yu = self.kernel(self._X, self._Z, *theta)
        K_uu = self.kernel(self._Z, self._Z, *theta) + self.noise * np.eye(self._Z.shape[0])
        Lambda = np.where(np.eye(self._X.shape[0]),K_yy - K_yu @ np.linalg.inv(K_uu) @ K_yu.T,0)
        Lambda_inv = np.linalg.inv(Lambda)
        return Lambda_inv - Lambda_inv @ K_yu @ np.linalg.inv(K_uu) @ K_yu.T @ Lambda_inv

    def C_N(self, theta):
        K_yy = self.kernel(self._X, self._X, *theta) + self.noise * np.eye(self._X.shape[0])
        K_yu = self.kernel(self._X, self._Z, *theta)
        K_uu = self.kernel(self._Z, self._Z, *theta) + self.noise * np.eye(self._Z.shape[0])
        Lambda = np.where(np.eye(self._X.shape[0]),K_yy - K_yu @ np.linalg.inv(K_uu) @ K_yu.T,0)
        return K_yu @  np.linalg.inv(K_uu) @ K_yu.T + Lambda + self.noise * np.eye(self._X.shape[0])

    def fit(self, X, Z):
        self._X = X
        self._Z = Z
        self._y = self.f(X)

        def marginal_likelihood(theta):
            return np.sum(np.log(np.diagonal(np.linalg.cholesky(self.C_N(theta))))) + \
                0.5 * self._y.T @ self.C_N_inv(theta) @ self._y + \
                0.5 * self._X.shape[0] * np.log(2 * np.pi)

        res = minimize(marginal_likelihood, x0=self.theta, bounds=self.bound, method='L-BFGS-B')
        self.theta = res.x

    def predict(self, X_test):
        self._X_test = X_test
        C_N = self.kernel(self._X, self._X, *self.theta) + self.noise * np.eye(self._X.shape[0])
        k = self.kernel(self._X, self._X_test, *self.theta)
        c = self.kernel(self._X_test, self._X_test, *self.theta) + self.noise * np.eye(self._X_test.shape[0])
        C_N_inv = np.linalg.inv(C_N)
        m = k.T @ C_N_inv @ self._y
        cov = c - k.T @ C_N_inv @ k
        s = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), 0))
        self._m = m
        self._s = s
        return m, s

    def plot(self):
        plt.plot(self._X_test.flatten(), self._m.flatten(), label=r'$\mu$')
        plt.plot(self._X_test.flatten(), self.f(self._X_test).flatten(), label=r'$f(x)$')
        plt.scatter(self._X.flatten(), self._y.flatten(), label='train data', marker='x', color='black')
        plt.fill_between(self._X_test.flatten(), self._m.flatten() - 1.96 * self._s.flatten(),
                         self._m.flatten() + 1.96 * self._s.flatten(), alpha=0.2, label=r'$\mu\pm 2\sigma$')
        sns.rugplot(self._Z.flatten(), label='pseudo data')
        plt.legend()


if __name__ == '__main__':
    # X = np.random.uniform(low=-1, high=1, size=(10, 1))
    # X_test = np.linspace(-1, 1, 100).reshape(-1, 1)
    # gp = GP(test_1D, rbf, (1, 1), ((1e-5, None), (1e-5, None)), noise=1e-5)
    # gp.fit(X)
    # gp.predict(X_test)
    # gp.plot()

    plt.figure(figsize=(16,8))
    X = np.random.uniform(low=-1, high=1, size=(20, 1))
    X_test = np.linspace(-1, 1, 100).reshape(-1, 1)
    for i in range(6):
        plt.subplot(3, 2, i+1)
        sgp = SparseGP(test_1D, rbf, (1, 1), ((1e-5, None), (1e-5, None)),noise=1e-3)
        sgp.fit(X,np.random.choice(X.reshape(-1),(i+1)*3,replace=False).reshape(-1,1))
        sgp.predict(X_test)
        sgp.plot()
        plt.title(f'#pseudo data = {3*(i+1)}')
    plt.tight_layout()
    plt.show(dpi=600)
