import numpy as np
import copy
from utils import sigmoid


"""
RBM Class Structure
"""

class RBM:
    def __init__(self, p, q):
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(size=(p, q)) * np.sqrt(0.01)

    def entree_sortie_RBM(self, V):
        return sigmoid(V @ self.W + self.b) #Multiplication matricielle

    def sortie_entree_RBM(self, H):
        return sigmoid(H @ self.W.T + self.a)

    def train_RBM(self, X, learning_rate, len_batch, n_epochs):
        p, q = self.W.shape

        weights = []
        losses = []

        for i in range(n_epochs):

            np.random.shuffle(X)
            n = X.shape[0]
            for i_batch in range(0, n, len_batch):
                X_batch = X[i_batch:min(i_batch + len_batch, n), :]
                t_batch_i = X_batch.shape[0]

                #Contrastive-Divergence-1 algorithm to estimate the gradient

                V0 = copy.deepcopy(X_batch)
                pH_V0 = self.entree_sortie_RBM(V0)
                H0 = (np.random.rand(t_batch_i, q) < pH_V0) * 1
                pV_H0 = self.sortie_entree_RBM(H0)
                V1 = (np.random.rand(t_batch_i, p) < pV_H0) * 1
                pH_V1 = self.entree_sortie_RBM(V1)

                grad_a = np.sum(V0 - V1, axis=0)
                grad_b = np.sum(pH_V0 - pH_V1, axis=0)
                grad_W = V0.T @ pH_V0 - V1.T @ pH_V1

                self.a += learning_rate * grad_a
                self.b += learning_rate * grad_b
                self.W += learning_rate * grad_W

                weights.append(np.mean(self.W))

            # Reconstruction's loss
            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            loss = np.mean((X - X_rec) ** 2) #quadratic norm
            losses.append(loss)
            print("epoch " + str(i) + "/" + str(n_epochs) + " - Loss RBM : " + str(loss))


    def generer_image_RBM(self, nb_images, nb_iter):
        p, q = self.W.shape
        images = []
        for i in range(nb_images):  # Gibbs
            v = (np.random.rand(p) < 0.5) * 1
            for j in range(nb_iter):
                h = (np.random.rand(q) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(p) < self.sortie_entree_RBM(h)) * 1

            images.append(v)

        return images


if __name__ == "__main__":

    from loading_data import lire_alpha_digit
    from utils import plot_images

    X, size_img = lire_alpha_digit(caractere=['A'])

    """
    Training RBM 
    """

    p, q = size_img[0] * size_img[1], 100
    rbm = RBM(p, q)  # Instance of RBM
    rbm.train_RBM(X, learning_rate=10 ** (-2), len_batch=10, n_epochs=1000)

    generated_images_rbm = rbm.generer_image_RBM(nb_images=10, nb_iter=200)
    plot_images(generated_images_rbm, database='BinaryAlphaDigit')



