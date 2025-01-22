import numpy as np
import copy

from loading_data import lire_alpha_digit
from utils import sigmoid, plot_images

"""
RBM Class Structure
"""


class RBM:
    def __init__(self, p, q):
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(size=(p, q)) * np.sqrt(0.01)

    def entree_sortie_RBM(self, V):
        return sigmoid(V @ self.W + self.b)

    def sortie_entree_RBM(self, H):
        return sigmoid(H @ self.W.T + self.a)

    def train_RBM(self, X, learning_rate, len_batch, n_epochs, verbose=1):
        p, q = self.W.shape

        for i in range(n_epochs):

            np.random.shuffle(X)

            n = X.shape[0]
            for ith_batch in range(0, n, len_batch):
                X_batch = X[ith_batch:min(ith_batch + len_batch, n), :]
                t_batch_i = X_batch.shape[0]

                # Contrastive-Divergence-1 algorithm to estimate the gradient
                V0 = copy.deepcopy(X_batch)

                pH_V0 = self.entree_sortie_RBM(V0)
                # draw from pH_V0
                H0 = (np.random.rand(t_batch_i, q) < pH_V0) * 1

                pV_H0 = self.sortie_entree_RBM(H0)
                # draw from pV_H0
                V1 = (np.random.rand(t_batch_i, p) < pV_H0) * 1

                pH_V1 = self.entree_sortie_RBM(V1)

                grad_a = np.sum(V0 - V1, axis=0)
                grad_b = np.sum(pH_V0 - pH_V1, axis=0)
                grad_W = V0.T @ pH_V0 - V1.T @ pH_V1

                self.a += learning_rate * grad_a
                self.b += learning_rate * grad_b
                self.W += learning_rate * grad_W

            # Reconstruction's loss
            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            loss = np.mean((X - X_rec) ** 2)  # quadratic norm

            if i % 10 == 0 and verbose:  # verbose for progression bar
                print(f"Epoch {i + 1}/{n_epochs}, Loss: {loss:.4f}")

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


if __name__ == "main":
    X, size_img = lire_alpha_digit('data/binaryalphadigs.mat', caractere=['A'])

    nb_features = size_img[0] * size_img[1]
    p, q = nb_features, 100
    rbm = RBM(p, q)  # Instance of RBM

    rbm.train_RBM(X, learning_rate=10 ** (-2), len_batch=10, n_epochs=1000, verbose=1)

    generated_images = rbm.generer_image_RBM(nb_images=10, nb_iter=200)
    plot_images(generated_images, size_img)