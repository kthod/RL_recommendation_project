from scipy.stats import binom

# Parameters
N = 200
k = 10
p = 4/25

# Calculate the cumulative probability P(X <= 10)
prob = binom.cdf(k, N, p)
print(prob)
import numpy as np

v = np.array([3, 7, 5, 2, 9, 1])
threshold = 5
p = np.where(v >= threshold, 1, 0)
print(p)
