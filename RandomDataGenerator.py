import numpy as np
import argparse
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from data_generator import GaussianNoise


def polynomial_basis_linear_model_data_generator(M, a, w, X):
    """
    t = y(x, w) + e, where e ~ N(0,a)
    input:
        M: the basis number
        a: variance
        w: parameters shape (M, 1)
        X: N data points
    return:
        t: target
    """

    phi = np.array([X**i for i in range(M)]) # shape = (M, N)
    return np.dot(np.transpose(w), phi)

def plot_gaussian_distribution(mu, sigma, data_num=10000, plot=True):
    # Generate data
    X = np.array([GaussianNoise.boxmuller(mu, sigma) for i in range(data_num)])

    # Histogram
    X_min = mu - 4*sigma
    X_max = mu + 4*sigma

    n, bins, patches = plt.hist(X, 100, normed=1, facecolor='blue', alpha=0.75)

    # plot the 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Data')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ data:}\ \mu=%.2f,\ \sigma=%.2f$' % (mu, sigma))
    plt.axis([X_min, X_max, 0, 1])
    plt.grid(True)

    if plot:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mean', type=float, help='specify the mean for Gaussian Noise', default=0.0)
    parser.add_argument('-v', '--variance', type=float, help='specify the variance for Gaussian Noise', default=1.0)
    parser.add_argument('-w', '--weights', type=str, help='specify the weights of data points', default="1,2")
    parser.add_argument('-M', '--order', type=int, help='specify the weights of data points', default=2)
    parser.add_argument('--num', type=int, help='specify the number of data points', default=100)
    args = parser.parse_args()

    # 1 (a)
    plot_gaussian_distribution(args.mean, args.variance)

    # 1 (b)
    W = np.array(args.weights.split(','), dtype=float)
    assert args.order == len(W)

    X = np.linspace(-10.0, 10.0, args.num)
    Y = polynomial_basis_linear_model_data_generator(args.order, args.variance, W, X)

    # plot
    fig = plt.figure()

    # draw function
    ax1 = fig.add_subplot(111)
    ax1.plot(X, Y, color="red")

    e = np.array([GaussianNoise.boxmuller(0, args.variance) for i in range(args.num)])
    T = Y + e

    # draw data
    ax1.scatter(X, T, s=5)
    plt.show()
