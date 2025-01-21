import numpy as np
import matplotlib.pyplot as plt

"""
Useful functions
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_images(X,database='BinaryAlphaDigit') :

    if database == 'BinaryAlphaDigit' :
        for image in X:
            image = image.reshape(320)
            plt.imshow(image, cmap='gray')
            plt.show()

    if database == 'MNIST' :
        for image in X:
            image = image.reshape((28,28))
            plt.imshow(image, cmap='gray')
            plt.show()
