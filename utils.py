import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
Useful functions
"""
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Limite les valeurs de x pour éviter l'overflow

    return 1 / (1 + np.exp(-x))

def plot_images(X,database='BinaryAlphaDigit') :

    if database == 'BinaryAlphaDigit' :
        for image in X:
            image = image.reshape((20,16))
            plt.imshow(image, cmap='gray')
            plt.show()

    if database == 'MNIST' :
        for image in X:
            image = image.reshape((28,28))
            plt.imshow(image, cmap='gray')
            plt.show()

def calcul_softmax(X):
    exp_x = np.exp(X - np.max(X, axis=1,keepdims=True))
    sum_exp_x = np.sum(exp_x, axis=1,keepdims=True)
    proba_sortie = exp_x / sum_exp_x
    return proba_sortie


def calcul_entropie_croisee(y_pred, y_batch) :
    y_batch = y_batch.to_numpy()  # Because data is very sparse

    loss = -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))

    return loss

def binarisation(y) :

    return pd.get_dummies(y, sparse=False)




