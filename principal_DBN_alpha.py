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
            self.rbms[l].train_RBM(X, learning_rate, len_batch, n_epochs)
            X = self.rbms[l].entree_sortie_RBM(X)


    def generer_image_DBN(self,nb_images, nb_iter):
        images = []

        for i in range(nb_images):

            #Gibbs sur la derniere couche cachée
            top_rbm = self.rbms[-1]
            p_top_rbm, q_top_rbm = top_rbm.W.shape
            # Initialisation aléatoire de la couche visible
            v = (np.random.rand(p_top_rbm) < 0.5) * 1

            for _ in range(nb_iter):
                h = (np.random.rand(q_top_rbm) < top_rbm.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(p_top_rbm) < top_rbm.sortie_entree_RBM(h)) * 1

            # Propagation vers l'arrière (caché -> visible)
            top_sample = v
            for rbm in reversed(self.rbms[:-1]):
                p_rbm, q_rbm = rbm.W.shape
                top_sample = (np.random.rand(p_rbm) < rbm.sortie_entree_RBM(top_sample)) * 1

            images.append(top_sample)

        return images


if __name__ == "__main__":

    """
    Training DBN
    """
    from loading_data import lire_alpha_digit
    from utils import plot_images

    X, size_img = lire_alpha_digit(caractere=['A'])


    dbn_size = [320, 200, 100]
    dbn = DBN(dbn_size) #Instance of RBM
    dbn.train_DBN(X, learning_rate=10**(-2), len_batch=10, n_epochs=500)

    generated_images = dbn.generer_image_DBN(nb_images=10, nb_iter=200)
    plot_images(generated_images, database='BinaryAlphaDigit')



