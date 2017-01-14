import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# generate sample data
x_data = np.linspace(-math.pi, math.pi, 30)
y_data = np.sin(x_data) + np.random.normal(0.0, 0.1, x_data.size)

plt.plot(x_data, y_data, 'o')
plt.show()

mean_est = tf.Variable(0.0)
# Use squared exponential covariance matrix
x_rows, x_cols = tf.meshgrid(x_data, x_data)
# Covariance defined as $exp(-0.5*(x_i-x_j)^2/l^2)$ where l is the length-scale
covariance_est = tf.exp(tf.scalar_mul(-0.5, tf.squared_difference(x_rows, x_cols)))

sess = tf.Session()
print(sess.run(covariance_est))


