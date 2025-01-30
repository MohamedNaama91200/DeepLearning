import copy
import numpy as np
import os
from loading_data import lire_alpha_digit
from principal_RBM_alpha import RBM
from utils import plot_images,save_images

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

        tmp = X.copy()

        # Greedy layer wise procedure
        for l in range(len(self.dbn_size) - 1):
            if verbose:
                print(f"Train RBM {l + 1}/{len(self.dbn_size) - 1}\t")
            self.rbms[l].train_RBM(tmp, learning_rate, len_batch, n_epochs, verbose)
            tmp = self.rbms[l].entree_sortie_RBM(tmp)

    def generer_image_DBN(self, nb_images, nb_iter):
        images = []

        for i in range(nb_images):

            # Gibbs sur la derniere couche cachée
            top_rbm = self.rbms[-1]
            p, q = top_rbm.W.shape

            # Initialisation aléatoire de la couche visible
            v = (np.random.rand(p) < 0.5) * 1

            for _ in range(nb_iter):
                h = (np.random.rand(q) < top_rbm.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(p) < top_rbm.sortie_entree_RBM(h)) * 1

            # Propagation vers l'arrière (caché -> visible)
            h = v
            for rbm in reversed(self.rbms[:-1]):
                p, q = rbm.W.shape
                h = (np.random.rand(p) < rbm.sortie_entree_RBM(h)) * 1

            images.append(h)

        return images

    @staticmethod
    def test_dbn_hyperparameters():
        os.makedirs("results/dbn/hyperparameters", exist_ok=True)

        # Paramètres à tester
        architectures = [
            [320, 100],          # 1 couche
            [320, 200, 100],     # 2 couches
            [320, 400, 200, 100] # 3 couches
        ]
        character_sets = [['A'], ['A', 'B'], ['A', 'B', 'C']]

        for arch in architectures:
            for chars in character_sets:
                X, size_img = lire_alpha_digit(caractere=chars)

                dbn = DBN(arch)
                dbn.train_DBN(X, learning_rate=1e-2, len_batch=10, n_epochs=1000)

                # Génération des images
                images = dbn.generer_image_DBN(nb_images=10, nb_iter=500)

                filename = f"results/dbn/hyperparameters/layers_{len(arch) - 1}_chars_{'-'.join(chars)}.png"

                save_images(images, size_img, filename=filename)


if __name__ == "__main__":
    DBN.test_dbn_hyperparameters()



