import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_sample(mean, cov_matrix):
    '''generate_sample: Generate sample function output from a mean and covariance matrix.'''
    cholesky_decomp = tf.cholesky(cov_matrix)
    cov_shape = tf.shape(cov_matrix)
    result_shape = [cov_shape[0], 1]
    uniform_gaussian_distribution = tf.random_normal(result_shape, mean=0.0, stddev=1.0,  \
        dtype=tf.float64)
    return mean + tf.matmul(cholesky_decomp, uniform_gaussian_distribution)

def solve_posterior(x_data, y_data, cov_matrix, sigma, test_data):
    '''solve_posterior: Generate the mean, variance and log marginal likelihood from
    sample data.'''
    cholesky_decomp = tf.cholesky(cov_matrix + math.pow(sigma, 2)*tf.eye(tf.shape(cov_matrix)[0], dtype=tf.float64))
    alpha = tf.cholesky_solve(cholesky_decomp, y_data)
    star_X_rows, star_X_cols = tf.meshgrid(x_data, test_data)
    K_star_X = tf.exp(tf.scalar_mul(-0.5, 
        tf.squared_difference(star_X_cols, star_X_rows)/length_scale))
    mean = tf.matmul(K_star_X, alpha)
    star_rows, star_cols = tf.meshgrid(test_data, test_data)
    K_star_star = tf.exp(tf.scalar_mul(-0.5,
        tf.squared_difference(star_cols, star_rows)/length_scale))
    X_star_rows, X_star_cols = tf.meshgrid(test_data, x_data)
    K_X_star = tf.exp(tf.scalar_mul(-0.5,
        tf.squared_difference(X_star_cols, X_star_rows)/length_scale))
    variance = K_star_star - tf.matmul(K_star_X, tf.cholesky_solve(cholesky_decomp, K_X_star))
    log_marg_likelihood = -0.5*tf.transpose(y_data)*alpha \
        - tf.reduce_sum(tf.log(tf.diag_part(cholesky_decomp))) \
        - (x_data.size / 2) * math.log(math.pi)
    return mean, variance, log_marg_likelihood

if __name__ == "__main__":
    # generate sample data
    x_data = np.random.rand(10) * 2 * math.pi - math.pi
    x_data = np.sort(x_data)
    y_data = np.sin(x_data) + np.random.normal(0.0, 0.1, x_data.size)

    mean_est = 0.0
    length_scale = 1.5
    # Use squared exponential covariance matrix
    # Covariance defined as $exp(-0.5*(x_i-x_j)^2/l^2)$ where l is the length-scale
    x_rows, x_cols = tf.meshgrid(x_data, x_data)
    covariance_est = tf.exp(tf.scalar_mul(-0.5, \
        tf.squared_difference(x_cols, x_rows)/length_scale))

    sess = tf.Session()

    # print prior samples
    num_samples = 0
    while num_samples < 5:
        prior_sample = sess.run(generate_sample(mean_est, covariance_est))
        plt.plot(x_data, prior_sample)
        plt.title('Prior Samples')
        num_samples = num_samples + 1
    
    plt.show()

    x_test = np.linspace(-math.pi - 0.5, math.pi + 0.5, 100)

    mean, variance, log_marg_likelihood = sess.run(solve_posterior(x_data, 
        tf.reshape(y_data, [y_data.size, 1]), covariance_est, 0.1, x_test))
    mean = mean.flatten()

    variance_diag = np.diagonal(variance)

    mean_plus_variance = mean + 2*variance_diag
    mean_minus_variance = mean - 2*variance_diag

    plt.plot(x_data, y_data, 'o')
    plt.plot(x_test, mean)
    plt.fill_between(x_test, mean_minus_variance, mean_plus_variance)
    plt.show()


