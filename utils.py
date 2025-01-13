import numpy as np
import matplotlib.pyplot as plt

"""
Useful functions
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_images(X,size_img) :
    for image in X:
        image = image.reshape(size_img)
        plt.imshow(image, cmap='gray')
        plt.show()
