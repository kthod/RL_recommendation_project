import numpy as np

# K = 5
# N = 2

# # Create a random KxK matrix
# matrix = np.random.rand(K, K)
# print(matrix)
# # Find the indices of the N max values in each row
# indices = np.argpartition(matrix, -N, axis=1)[:, -N:]
# print(indices)
# # Construct the KxN matrix using the indices
# result = np.take_along_axis(matrix, indices, axis=1)

# print(result)

from math import comb

def k_to_combination(point,k, N, K, offset):
   
    if(N == 1):
        point.append(k+offset)
        return point
    lower = 1
    upper = comb(K-1,N-1)
    i=0
    
    while k+1 > upper:
        i += 1
        lower = upper + 1
        upper = upper + comb(K-1-i,N-1)
    # if(i+offset == state):
    #     offset+=1
    point.append(i+offset)
    #print(point)
    return k_to_combination(point,k-lower+1,N-1,K-i-1,i+offset+1)

# Example usage:
N = 2
K = 8
for i in range(comb(K,N)):
    point = []
    point = k_to_combination(point,i, N, K, 0)  # Output: [0, 1, 2], [0, 1, 3], ...
    print(point)
    



