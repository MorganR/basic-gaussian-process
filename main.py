import math
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def generate_sample(mean, cov_matrix):
    '''generate_sample: Generate sample function output from a mean and covariance matrix.'''
    cholesky_decomp = linalg.cholesky(cov_matrix)
    cov_shape = cov_matrix.shape
    uniform_gaussian_distribution = np.random.normal(loc=0.0, scale=1.0, size=cov_shape[0])
    return mean + np.matmul(cholesky_decomp, uniform_gaussian_distribution)

def get_covariance_matrix(x, length_scale):
    # Use squared exponential covariance matrix
    # Covariance defined as $exp(-0.5*(x_i-x_j)^2/l^2)$ where l is the length-scale
    x_rows, x_cols = np.meshgrid(x, x)
    return np.exp((-0.5 * (x_cols - x_rows)**2)/length_scale**2)

def solve_posterior(x_data, y_data, cov_matrix, sigma, test_data):
    '''solve_posterior: Generate the mean, variance and log marginal likelihood from
    sample data.'''
    cholesky_decomp = linalg.cho_factor(cov_matrix + (sigma**2)*np.eye(cov_matrix.shape[0]))
    alpha = linalg.cho_solve(cholesky_decomp, y_data)
    star_X_rows, star_X_cols = np.meshgrid(x_data, test_data)
    K_star_X = np.exp(-0.5*
        (star_X_cols-star_X_rows)**2/(length_scale**2))
    mean = np.matmul(K_star_X, alpha)
    star_rows, star_cols = np.meshgrid(test_data, test_data)
    K_star_star = np.exp(-0.5*
        (star_cols - star_rows)**2/length_scale**2)
    X_star_rows, X_star_cols = np.meshgrid(test_data, x_data)
    K_X_star = np.exp(-0.5*
        (X_star_cols-X_star_rows)**2/length_scale**2)
    variance = K_star_star - np.matmul(K_star_X, linalg.cho_solve(cholesky_decomp, K_X_star))
    log_marg_likelihood = -0.5*np.matmul(y_data.T,alpha) \
        - np.sum(np.log(np.diagonal(cholesky_decomp[0]))) \
        - (x_data.size / 2) * math.log(math.pi)
    return mean, variance, log_marg_likelihood

def perform_regression(x_data, y_data, x_min, x_max, mean_est, length_scale):
    # Estimated a covariance matrix for the givent data
    covariance_est = get_covariance_matrix(x_data, length_scale)

    x_test = np.linspace(x_min, x_max, 100)

    mean, variance, log_marg_likelihood = solve_posterior(x_data,
        y_data, covariance_est, 0.1, x_test)
    mean = mean.flatten()
    print('Log marginal likelihood: ', log_marg_likelihood)

    variance_diag = np.diagonal(variance)

    mean_plus_variance = mean + 2*variance_diag
    mean_minus_variance = mean - 2*variance_diag

    plt.plot(x_data, y_data, 'o')
    plt.plot(x_test, mean)
    plt.fill_between(x_test, mean_minus_variance, mean_plus_variance, color='grey')
    plt.title('GP Regression with Length Scale = {}'.format(length_scale))
    plt.xlabel('x')
    plt.ylabel('y')
    fig = plt.gcf()
    fig.savefig('results-l-{:.1f}.eps'.format(length_scale))
    plt.show()

if __name__ == "__main__":
    # generate sample data
    x_min = -math.pi
    x_max = math.pi
    x_data = np.random.rand(10) * (x_max - x_min) - (x_max - x_min)/2
    x_data = np.sort(x_data)
    y_data = np.random.normal(loc=0.0, scale=1.0, size=x_data.size)

    mean_est = 0.0
    length_scale = 1

    # Generate samples from the prior
    x_sample = np.linspace(x_min, x_max, 20)
    covariance_sample = get_covariance_matrix(x_sample, length_scale)    

    plt.title('Sample Values of the Squared\nExponential Covariance Matrix')
    plt.imshow(covariance_sample)
    plt.colorbar()
    fig = plt.gcf()
    fig.savefig('cov-matrix.eps')
    plt.show()

    # print prior samples
    num_samples = 0
    while num_samples < 5:
        prior_sample = generate_sample(mean_est, covariance_sample)
        plt.plot(x_sample, prior_sample)
        num_samples = num_samples + 1
    plt.title('Samples from the Prior')
    plt.xlabel('x')
    plt.ylabel('y')
    fig = plt.gcf()
    fig.savefig('prior.eps')
    plt.show()

    perform_regression(x_data, y_data, x_min, x_max, mean_est, length_scale)
    perform_regression(x_data, y_data, x_min, x_max, mean_est, 0.5)
    perform_regression(x_data, y_data, x_min, x_max, mean_est, 1.5)
    