


# a = [[2, 4], [2, 2], [3, 3], [1, 4], [0, 4], [2, 3], [1, 2], [3, 4], [1, 3], [0, 3], [0, 2], [0, 0], [0, 1], [1, 1]]


# print(sorted(a))

# b = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 1], [1, 2], [1, 3], [1, 4], [2, 2], [2, 3], [2, 4], [3, 3], [3, 4]]

# print(sorted(b))

import numpy as np
import ml_dtypes


a = np.array([[1526, 4.3]], dtype=ml_dtypes.bfloat16)
print(a)


b = np.array([[774.492, 1526]], dtype=np.float64)

b = b.astype(ml_dtypes.bfloat16)

print(b)
