from loading_data import lire_alpha_digit
from utils import plot_images
import numpy as np

"""
Main Code for Deep Neural Networks 
"""
X, size_img = lire_alpha_digit(caractere=['A'])

#Random images from the BinAlphaDigit dataset
for _ in range(5):
    i = np.random.choice(X.shape[0])
    plot_images([X[i]],database='BinaryAlphaDigit')

