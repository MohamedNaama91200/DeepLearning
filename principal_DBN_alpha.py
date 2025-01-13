import numpy as np
import copy
import matplotlib.pyplot as plt
from utils import sigmoid
from principal_RBM_alpha import RBM

"""
DBN Class Structure
"""

class DBN:
    def __init__(self, dbn_size):
        """
        :param dbn_size: List [] of numbers of neurons per layer.
        """
        self.dbn_size = dbn_size
        self.rbms = []  # List to store RBMs

        # Initialize RBMs for each pair of consecutive layers
        for l in range(len(dbn_size) - 1):
            p = dbn_size[l]
            q = dbn_size[l + 1]
            rbm = RBM(p, q)
            self.rbms.append(rbm)

    def train_DBN(self, X, learning_rate, len_batch, n_epochs,dbn_size):

        #Greedy layer wise procedure

        for l in range(len(dbn_size) - 1) :
            self.rbms[l].train_RBM(X, learning_rate, len_batch, n_epochs,verbose=1)
            X = self.rbms[l].entree_sortie_RBM(X)


    def generer_image_RBM(self, nb_images, nb_iter, size_img):
        p, q = self.W.shape
        images = []
        for i in range(nb_images):  # Gibbs
            v = (np.random.rand(p) < 0.5) * 1
            for j in range(nb_iter):
                h = (np.random.rand(q) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(p) < self.sortie_entree_RBM(h)) * 1
            v = v.reshape(size_img)
            images.append(v)
        return images


