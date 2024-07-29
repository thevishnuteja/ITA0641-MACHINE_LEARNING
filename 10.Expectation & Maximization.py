import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# Generate sample data
np.random.seed(42)
data = np.vstack([np.random.multivariate_normal(mean, 0.1*np.eye(2), 100) 
                  for mean in [(0, 0), (3, 3), (6, 0)]])
df = pd.DataFrame(data, columns=['x', 'y'])

# EM Algorithm for GMM
class EM_GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.means_ = X[np.random.choice(n_samples, self.n_components, False)]
        self.covariances_ = np.array([np.eye(n_features)] * self.n_components)
        self.weights_ = np.ones(self.n_components) / self.n_components

    def e_step(self, X):
        self.resp_ = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            self.resp_[:, k] = self.weights_[k] * multivariate_normal(self.means_[k], self.covariances_[k]).pdf(X)
        self.resp_ /= self.resp_.sum(axis=1, keepdims=True)

    def m_step(self, X):
        Nk = self.resp_.sum(axis=0)
        self.weights_ = Nk / X.shape[0]
        self.means_ = np.dot(self.resp_.T, X) / Nk[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(self.resp_[:, k] * diff.T, diff) / Nk[k]

    def compute_log_likelihood(self, X):
        log_likelihood = np.log(
            np.array([self.weights_[k] * multivariate_normal(self.means_[k], self.covariances_[k]).pdf(X) 
            for k in range(self.n_components)]).sum(axis=0)
        )
        return np.mean(log_likelihood)

    def fit(self, X):
        self.initialize_parameters(X)
        log_likelihood = self.compute_log_likelihood(X)
        for i in range(self.max_iter):
            prev_log_likelihood = log_likelihood
            self.e_step(X)
            self.m_step(X)
            log_likelihood = self.compute_log_likelihood(X)
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
        return self

    def predict_proba(self, X):
        self.e_step(X)
        return self.resp_

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# Applying EM Algorithm to the Data
em_gmm = EM_GMM(n_components=3)
em_gmm.fit(df.values)
df['cluster'] = em_gmm.predict(df.values)

print("Means:\n", em_gmm.means_)
print("Covariances:\n", em_gmm.covariances_)
print("Weights:\n", em_gmm.weights_)
print("Data with cluster assignments:\n", df.head())
