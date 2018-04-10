import numpy as np


class GaussianNoise:
    @staticmethod
    def boxmuller(mean, var):
        # implement Box–Muller method (1958)
        # two independent random numbers U and V distributed uniformly on (0,1)
        U = np.random.uniform()
        V = np.random.uniform()
        x_1 = np.sqrt((-2)*np.log(U)) * np.cos(2*np.pi*V)
        x_2 = np.sqrt((-2)*np.log(U)) * np.sin(2*np.pi*V)

        return mean + x_1 * var

    @staticmethod
    def marsaglia(mean, var):
        d = 1.0
        while d >= 1.0:
            x1 = 2.0 * np.random.uniform() - 1.0
            x2 = 2.0 * np.random.uniform() - 1.0
            d = x1*x1 + x2*x2
        d = np.sqrt((-2.0*np.log(d))/d)

        return mean + x1 * d * var
