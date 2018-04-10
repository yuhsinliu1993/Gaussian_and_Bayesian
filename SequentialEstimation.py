# Sequential estimate the mean and variance from the data given from the univariate gaussian data generator

import argparse, sys
import numpy as np
from data_generator import GaussianNoise

epsilon = 1e-5
sys.setrecursionlimit(10000)

def sequential_estimation(mu, sigma, existingAggregate):
    count, mean, M2 = existingAggregate

    if np.abs(mean - mu) < epsilon and count>=1:
        return
    else:
        count += 1
        print("---------------- {} iteration ----------------".format(count))

        new_x = GaussianNoise.boxmuller(mu, sigma)
        print("[*] new data point: {}".format(new_x))

        delta = new_x - mean
        mean = mean + delta / count
        delta2 = new_x - mean
        M2 += delta * delta2

        if count < 2:
            variance = float('nan')
        else:
            variance = M2/(count-1)

        print('Mean: {}  Variance: {}'.format(mean, variance))

        existingAggregate = (count, mean, M2)
        sequential_estimation(mu, sigma, existingAggregate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mean', type=float, help='specify the mean for Gaussian Noise', default=0.0)
    parser.add_argument('-v', '--variance', type=float, help='specify the variance for Gaussian Noise', default=1.0)
    args = parser.parse_args()

    existingAggregate = (0, 0.0, 0.0)
    sequential_estimation(mu=args.mean, sigma=args.variance, existingAggregate=existingAggregate)
