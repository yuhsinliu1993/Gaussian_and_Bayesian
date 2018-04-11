# Baysian Linear regression
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_generator import GaussianNoise
from RandomDataGenerator import polynomial_basis_linear_model_data_generator


def gaussian_probability(x, mean, variance):
    """
    Calculate Gaussian probability.
    """
    stdev = np.sqrt(variance)
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev, 2))))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

def coor_probability(coor_X, coor_Y, M, noise_precision, mu, precision):
    """
    Parameters:
    -----------
    coor_X: (data_num, data_num)  the x-direction coordinates
    coor_Y: (data_num, data_num)  the y-direction coordinates
    M: order
    noise_precision: alpha, a scalar
    mu, precision: the current Gaussian prior parameters

    Returns:
    --------
    the probability for every point (x, y) in (coor_X, coor_Y)
    """
    p_mu = np.zeros(coor_X.shape)
    p_var = np.zeros(coor_X.shape)
    p_y = np.zeros(coor_X.shape[0])

    for i in range(coor_X.shape[0]):
        p_mu[:, i], p_var[:, i], p_y[i] = predictive_distribution(M, coor_X[0, i], noise_precision, mu, precision)

    return gaussian_probability(coor_Y, p_mu, p_var)


def predictive_distribution(M, x, noise_precision, mu, precision):
    """
    Calculate predictive distribution, and get the parameters mu & variance given the data (x, t)
    p(t_new | x, t, a, b) = N(t | mu_T x A, new_var)

    mu = A_T x mu
    variance = 1/noise_precision + A_T x inverse(precision) x A
    """

    A = np.array([x**j for j in range(M)]).reshape((M, 1))  # (1, M)

    mu = np.dot(np.transpose(A), mu)  # (1, M) x (M, 1)
    var = 1 / noise_precision + np.dot(np.dot(np.transpose(A), np.linalg.inv(precision)), A)
    y = GaussianNoise.boxmuller(mu, var)

    return mu, var, y


def posterior_distribution(x, t, M, noise_precision, prior_mu, prior_precision):
    """
    Caculate the posterior distribution given a new data (x, t), and return the update mean and precision
      Prior:      multivariate Gaussian
    x Likelihood: univariate   Gaussian
    ------------------------------------
      Posterior:  multivariate Gaussian

    Parameters:
    -----------
    (x, t): a data point
    M: the order
    noise_precision: alpha, a scalar

    A: design matrix, shape (N, 1)   [1 x x^2 x^3 ...]
    b: prior precision matrix (bI)   (M, M)

    Returns:
    --------
    new precision = prior_precision + noise_precision x A_T x A
    new mu = new_variance x (prior_precision x mu + noise_precision x A_T x y)
    """
    A = np.array([x ** i for i in range(M)]).reshape((1, M)) # (M, 1)

    new_precision = prior_precision + noise_precision * np.dot(np.transpose(A), A)
    new_mu = np.dot(np.linalg.inv(new_precision), noise_precision * t * np.transpose(A) + np.dot(prior_precision, prior_mu))

    return new_mu, new_precision



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mean', type=float, help='specify the mean for Gaussian Noise', default=0.0)
    parser.add_argument('-g', '--gaussian_noise_precision', type=float, help='specify the precision for Gaussian Noise', default=0.2)
    parser.add_argument('-p', '--precision', type=float, help='specify the precision for initial prior', default=0.04)
    parser.add_argument('-w', '--weights', type=str, help='specify the weights of data points', default="1,2")
    parser.add_argument('-M', '--order', type=int, help='specify the weights of data points', default=2)
    parser.add_argument('--num', type=int, help='specify the number of data points', default=21)
    parser.add_argument('--plot', action='store_true', help='Whether plot the prior distribution or not', default=True)
    args = parser.parse_args()

    W = np.array(args.weights.split(','), dtype=float)
    assert args.order == len(W)

    # Step 1: generator data from polynomial basis linear model (M, a, w, X) + Gaussian Noise with 0 mean, precision a
    X = np.linspace(-10, 10, args.num)
    y = np.zeros(len(X))
    _, t = polynomial_basis_linear_model_data_generator(X, W, 0, args.gaussian_noise_precision)


    # Step2: update the prior, calculate the parameters of predictive distribution,
    #        repeat until the posterior probability converges
    # initialization: mu = 0, precision = bI   p(w | 0, bI)
    mu = np.zeros((args.order, 1))
    precision = args.precision * np.identity(args.order)

    print("-------------------- Posterior: mean & precision (No data) --------------------")
    print("mean:\n{}".format(mu))
    print("precision:\n{}".format(precision))

    for i in range(args.num):
        # get a data point (x, y)
        index = int(np.random.random() * len(X))

        # calculate the posterior p(w|t, X, noise_precision), and update the prior given the new data point (x, t)
        new_mu, new_precision = posterior_distribution(X[index], t[index], args.order, args.gaussian_noise_precision, mu, precision)

        if i % 2 == 0:
            print("\n--------------------- {} iteration ---------------------".format(i))
            print("New Data (x, t): ({0:.4f}, {1:.4f})".format(X[index], t[index]))

            print("Posterior Distribution")
            print("mean:\n{}".format(new_mu))
            print("precision:\n{}\n".format(new_precision))

            p_mu, p_var, _ = predictive_distribution(args.order, X[index], args.gaussian_noise_precision, mu, precision)
            print("redictive Distribution")
            print("mean:\n{}".format(p_mu[0][0]))
            print("precision:\n{}".format(np.linalg.inv(p_var)[0][0]))

        mu, precision = new_mu, new_precision

            # plt.scatter(X[index], t[index])

            # w0 = np.linspace(-10, 10, args.num)
            # w1 = np.linspace(-10, 10, args.num)
            # coor_X, coor_Y = np.meshgrid(w0, w1)
            #
            # p = coor_probability(coor_X, coor_Y, args.order, args.gaussian_noise_precision, mu, precision)
            # plt.imshow(p, origin="lower", extent=[-10, 10, -10, 10])
            # plt.colorbar()
            # plt.show()
