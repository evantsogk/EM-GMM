import numpy as np


# the EM with gaussian mixture model
class EM:
    def __init__(self, x, k):
        self.x = x
        self.K = k

        np.random.seed(0)

        self.p = np.random.rand(self.K, 1)  # priori probabilities
        self.m = np.random.rand(self.K, x.shape[1])  # means
        self.s_square = np.random.rand(self.K, 1)  # variance

        self.g = np.ndarray((self.x.shape[0], self.K))  # posteriori probabilities
        self.likelihood = 0
        self.old_likelihood = 0

    # the EM algorithm
    def ml_em(self, max_iter):

        for i in range(1, max_iter+1):

            self.expectation()
            self.maximization()

            self.log_likelihood()

            # ΕΜ convergence
            print(str(i) + ".", self.likelihood)
            if self.likelihood - self.old_likelihood < 0:
                print("Error found")
                return
            elif self.likelihood - self.old_likelihood < 1e-6:
                print("Converged")
                break

        return self.m, self.g

    # the expectation step, calculates g with numerical stability
    def expectation(self):

        f = []
        for k in range(self.K):
            fk = np.log(self.p[k])
            for d in range(self.x.shape[1]):
                fk = fk - (self.x[:, d] - self.m[k, d])**2 / (2*self.s_square[k]) - \
                     np.log(2*self.p[k]*self.s_square[k])/2

            f.append(fk)

        f = np.array(f).T
        m = np.max(f, axis=1)
        f = np.exp(f - m.reshape(m.shape[0], -1))

        self.g = f / (np.sum(f, axis=1).reshape(f.shape[0], -1))

    # the maximization step
    def maximization(self):

        sum_g = np.sum(self.g, axis=0)
        sum_g = sum_g.reshape(sum_g.shape[0], -1)

        # m new
        self.m = (np.dot(self.g.T, self.x)) / sum_g

        # s_square new
        self.s_square = []
        for k in range(self.K):
            s = 0
            for d in range(self.x.shape[1]):
                s += np.dot(self.g[:, k].T, (self.x[:, d] - self.m[k, d])**2)

            self.s_square.append(s)

        self.s_square = np.array(self.s_square)
        self.s_square = self.s_square.reshape(self.s_square.shape[0], -1)

        self.s_square /= 3*sum_g

        # p new
        self.p = sum_g / self.x.shape[0]

    # calculates the log likelihood with numerical stability
    def log_likelihood(self):

        self.old_likelihood = self.likelihood

        f = []
        for k in range(self.K):
            fk = np.log(self.p[k])
            for d in range(self.x.shape[1]):
                fk = fk - (self.x[:, d] - self.m[k, d])**2 / (2*self.s_square[k]) - \
                     np.log(2*self.p[k]*self.s_square[k])/2

            f.append(fk)
        f = np.array(f).T

        m = np.max(f, axis=1)
        f = np.exp(f - m.reshape(m.shape[0], -1))

        self.likelihood = np.sum(np.sum(m) + np.sum(np.log(np.sum(f, axis=1).reshape(f.shape[0], -1))))
