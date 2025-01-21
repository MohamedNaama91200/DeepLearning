import numpy as np
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

    def train_DBN(self, X, learning_rate, len_batch, n_epochs):

        #Greedy layer wise procedure

        for l in range(len(self.dbn_size) - 1) :
            self.rbms[l].train_RBM(X, learning_rate, len_batch, n_epochs,verbose=1)
            X = self.rbms[l].entree_sortie_RBM(X)


    def generer_image_DBN(self, X,nb_images, nb_iter, size_img):
        p = self.dbn_size[0]  # Taille de la couche visible du premier RBM
        images = []

        for i in range(nb_images):
            # Initialisation aléatoire de la couche visible
            v = (np.random.rand(p) < 0.5) * 1

            # Gibbs sampling sur toutes les couches
            for _ in range(nb_iter):
                # Propagation vers l'avant (visible -> caché)
                h = v
                for rbm in self.rbms:
                    p_rbm, q_rbm = rbm.W.shape
                    h = (np.random.rand(q_rbm) < rbm.entree_sortie_RBM(h)) * 1

                # Propagation vers l'arrière (caché -> visible)
                v = h
                for rbm in reversed(self.rbms):
                    p_rbm, q_rbm = rbm.W.shape
                    v = (np.random.rand(p_rbm) < rbm.sortie_entree_RBM(v)) * 1

            v = v.reshape(size_img)
            images.append(v)

        return images


