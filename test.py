import numpy as np

array = np.array([[0, 1, 0, 1], [1, 1, 1, 1]])
array2 = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])

array = (array >= 0.5).astype(int)
length = len(array2)

accuracy = np.sum(array == array2) / (len(array2)*len(array2[0]))

print("Accuracy on benign test examples: {}%".format(accuracy * 100))
print('end')
