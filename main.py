import math
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def generate_sample(mean, cov_matrix):
    '''generate_sample: Generate sample function output from a mean and covariance matrix.'''
    cov_shape = cov_matrix.shape
    temp_cov_matrix = cov_matrix
    while not is_pos_def(temp_cov_matrix):
        temp_cov_matrix = temp_cov_matrix + 0.01*np.eye(cov_shape[0])
    cholesky_decomp = linalg.cholesky(temp_cov_matrix)
    uniform_gaussian_distribution = np.random.normal(loc=0.0, scale=1.0, size=cov_shape[0])
    return mean + np.matmul(cholesky_decomp, uniform_gaussian_distribution)

def calc_squared_exponential(x_cols, x_rows, length_scale):
    return np.exp((-0.5 * (x_cols - x_rows)**2)/length_scale**2)

def get_covariance_matrix(x, length_scale):
    # Use squared exponential covariance matrix
    # Covariance defined as $exp(-0.5*(x_i-x_j)^2/l^2)$ where l is the length-scale
    x_rows, x_cols = np.meshgrid(x, x)
    return calc_squared_exponential(x_cols, x_rows, length_scale)

def solve_posterior(x_data, y_data, cov_matrix, sigma, test_data):
    '''solve_posterior: Generate the mean, variance and log marginal likelihood from
    sample data.'''
    cholesky_decomp = linalg.cho_factor(cov_matrix + (sigma**2)*np.eye(cov_matrix.shape[0]))
    alpha = linalg.cho_solve(cholesky_decomp, y_data)
    star_X_rows, star_X_cols = np.meshgrid(x_data, test_data)
    K_star_X = calc_squared_exponential(star_X_cols, star_X_rows, length_scale)
    mean = np.matmul(K_star_X, alpha)
    star_rows, star_cols = np.meshgrid(test_data, test_data)
    K_star_star = calc_squared_exponential(star_cols, star_rows, length_scale)
    X_star_rows, X_star_cols = np.meshgrid(test_data, x_data)
    K_X_star = calc_squared_exponential(X_star_cols, X_star_rows, length_scale)
    variance = K_star_star - np.matmul(K_star_X, linalg.cho_solve(cholesky_decomp, K_X_star))
    log_marg_likelihood = -0.5*np.matmul(y_data.T,alpha) \
        - np.sum(np.log(np.diagonal(cholesky_decomp[0]))) \
        - (x_data.size / 2) * math.log(math.pi)
    return mean, variance, log_marg_likelihood

def perform_regression(x_data, y_data, x_min, x_max, mean_est, length_scale):
    # Estimated a covariance matrix for the givent data
    covariance_est = get_covariance_matrix(x_data, length_scale)

    x_test = np.linspace(x_min, x_max, 20)

    mean, variance, log_marg_likelihood = solve_posterior(x_data,
        y_data, covariance_est, 0.1, x_test)
    mean = mean.flatten()
    print('Log marginal likelihood: ', log_marg_likelihood)

    variance_diag = np.diagonal(variance)

    x_smooth = np.linspace(x_min, x_max, 200)
    for n in np.arange(0, 3):
        sample = generate_sample(mean, variance)
        smooth_sample = spline(xk=x_test, yk=sample, xnew=x_smooth)
        plt.plot(x_smooth, smooth_sample, label='Posterior Sample')
    
    plt.plot(x_data, y_data, 'o')
    smooth_mean = spline(xk=x_test, yk=mean, xnew=x_smooth)
    smooth_variance = spline(xk=x_test, yk=variance_diag, xnew=x_smooth)
    mean_plus_variance = smooth_mean + 2*smooth_variance
    mean_minus_variance = smooth_mean - 2*smooth_variance
    plt.plot(x_smooth, smooth_mean, label='Mean')
    plt.fill_between(x_smooth, mean_minus_variance, mean_plus_variance, color='grey')
    plt.title('GP Regression with Length Scale = {}'.format(length_scale))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim((-2.5, 2.5))
    plt.legend()
    fig = plt.gcf()
    fig.savefig('results-l-{:.1f}.eps'.format(length_scale))
    plt.show()

if __name__ == "__main__":
    # generate sample data
    x_min = -4
    x_max = 4
    x_data = np.random.rand(5) * (x_max - x_min) - (x_max - x_min)/2
    x_data = np.sort(x_data)
    y_data = np.random.normal(loc=0.0, scale=1.0, size=x_data.size)

    mean_est = 0.0
    length_scale = 1

    # Generate samples from the prior
    x_sample = np.linspace(x_min, x_max, 20)
    covariance_sample = get_covariance_matrix(x_sample, length_scale)    

    plt.title('Sample Values of the Squared\nExponential Covariance Matrix')
    plt.imshow(covariance_sample, cmap='Greys')
    plt.colorbar()
    fig = plt.gcf()
    fig.savefig('cov-matrix.eps')
    plt.show()

    # Plot prior samples
    num_samples = 0
    x_smooth = np.linspace(x_min, x_max, 100)
    while num_samples < 3:
        prior_sample = generate_sample(mean_est, covariance_sample)
        smooth_sample = spline(xk=x_sample, yk=prior_sample, xnew=x_smooth)
        plt.plot(x_smooth, smooth_sample)
        num_samples = num_samples + 1
    # Plot the 2-sigma region
    plt.fill_between(x_sample, mean_est-2, mean_est+2, color='grey')
    plt.title('Samples from the Prior')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim((-2.5, 2.5))
    fig = plt.gcf()
    fig.savefig('prior.eps')
    plt.show()

    perform_regression(x_data, y_data, x_min, x_max, mean_est, length_scale)
    perform_regression(x_data, y_data, x_min, x_max, mean_est, 0.7)
    perform_regression(x_data, y_data, x_min, x_max, mean_est, 1.3)
    