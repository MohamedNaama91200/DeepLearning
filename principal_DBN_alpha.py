import numpy as np

from loading_data import lire_alpha_digit
from principal_RBM_alpha import RBM
from utils import plot_images

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

    def train_DBN(self, X, learning_rate, len_batch, n_epochs, verbose=1):
        # Greedy layer wise procedure
        for l in range(len(self.dbn_size) - 1):
            if verbose:
                print(f"Train RBM {l + 1}/{len(self.dbn_size) - 1}\t")
            self.rbms[l].train_RBM(X, learning_rate, len_batch, n_epochs, verbose)
            X = self.rbms[l].entree_sortie_RBM(X)

    def generer_image_DBN(self, nb_images, nb_iter):

        p = self.dbn_size[0]  # Size of the visible layer of the first RBM
        images = []

        for i in range(nb_images):
            # Random initialization of the visible layer
            v = (np.random.rand(p) < 0.5) * 1

            # Gibbs sampling across all layers
            for _ in range(nb_iter):
                # Forward propagation (visible -> hidden)
                h = v
                for rbm in self.rbms:
                    p_rbm, q_rbm = rbm.W.shape
                    h = (np.random.rand(q_rbm) < rbm.entree_sortie_RBM(h)) * 1

                # Backward propagation (hidden -> visible)
                v = h
                for rbm in reversed(self.rbms):
                    p_rbm, q_rbm = rbm.W.shape
                    v = (np.random.rand(p_rbm) < rbm.sortie_entree_RBM(v)) * 1

            images.append(v)

        return images

if __name__ == "main":
    X, size_img = lire_alpha_digit('data/binaryalphadigs.mat', caractere=['A'])

    dbn_size = [320, 200, 100]
    dbn = DBN(dbn_size)  # Instance of DBN
    dbn.train_DBN(X, learning_rate=10 ** (-2), len_batch=10, n_epochs=1000, verbose=1)

    generated_images = dbn.generer_image_DBN(nb_images=10, nb_iter=200)
    plot_images(generated_images, size_img)


