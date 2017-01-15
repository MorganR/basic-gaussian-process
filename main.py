import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_sample(mean, cov_matrix):
    '''generate_sample: Generate sample function output from a mean and covariance matrix.'''
    cholesky_decomp = tf.cholesky(cov_matrix)
    cov_shape = tf.shape(cov_matrix)
    result_shape = [cov_shape[0], 1]
    uniform_gaussian_distribution = tf.random_normal(result_shape, mean=0.0, stddev=1.0, dtype=tf.float64)
    return mean + tf.matmul(cholesky_decomp, uniform_gaussian_distribution)

if __name__ == "__main__":
    # generate sample data
    x_data = np.linspace(-math.pi, math.pi, 10)
    y_data = np.sin(x_data) + np.random.normal(0.0, 0.1, x_data.size)

    plt.plot(x_data, y_data, 'o')
    plt.show()

    mean_est = 0.0
    # Use squared exponential covariance matrix
    x_rows, x_cols = tf.meshgrid(x_data, x_data)
    # Covariance defined as $exp(-0.5*(x_i-x_j)^2/l^2)$ where l is the length-scale
    covariance_est = tf.exp(tf.scalar_mul(-0.5, tf.squared_difference(x_cols, x_rows)/2))

    sess = tf.Session()
    
    # print prior samples
    num_samples = 0
    while (num_samples < 5):
        prior_sample = sess.run(generate_sample(mean_est, covariance_est))
        plt.plot(x_data, prior_sample)
        plt.title('Prior Samples')
        num_samples = num_samples + 1
    
    plt.show()



