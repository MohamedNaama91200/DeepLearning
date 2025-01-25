import numpy as np
import copy
import os
from loading_data import lire_alpha_digit
from utils import sigmoid, plot_images,save_images
import pandas as pd

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

        n_samples = X.shape[0]

        for epoch in range(n_epochs):

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]

            for ith_batch in range(0, n_samples, len_batch):
                X_batch = X_shuffled[ith_batch:ith_batch + len_batch]
                m = X_batch.shape[0]

                # Contrastive-Divergence-1 algorithm to estimate the gradient
                V0 = X_batch.copy()

                pH_V0 = self.entree_sortie_RBM(V0)
                # draw from pH_V0
                H0 = (np.random.rand(m, q) < pH_V0) * 1

                pV_H0 = self.sortie_entree_RBM(H0)
                # draw from pV_H0
                V1 = (np.random.rand(m, p) < pV_H0) * 1

                pH_V1 = self.entree_sortie_RBM(V1)

                grad_a = np.sum(V0 - V1, axis=0)
                grad_b = np.sum(pH_V0 - pH_V1, axis=0)
                grad_W = V0.T @ pH_V0 - V1.T @ pH_V1

                self.a += learning_rate * grad_a
                self.b += learning_rate * grad_b
                self.W += learning_rate * grad_W

            # Reconstruction's loss
            H = self.entree_sortie_RBM(X_shuffled)
            X_rec = self.sortie_entree_RBM(H)
            loss = np.mean((X_shuffled - X_rec) ** 2)  # quadratic norm

            if epoch % 10 == 0 and verbose:  # verbose for progression bar
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")

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

    @staticmethod
    def test_rbm_hyperparameters():
        os.makedirs("results/rbm/hyperparameters", exist_ok=True)
        hidden_units_list = [50, 100, 200, 300]
        caractere_list = [['A'], ['A', 'B'], ['A', 'B', 'C']]

        for caractere in caractere_list:
            X, size_img = lire_alpha_digit('data/binaryalphadigs.mat', caractere=caractere)
            p = size_img[0] * size_img[1]

            for q in hidden_units_list:
                rbm = RBM(p, q)
                rbm.train_RBM(X, learning_rate=1e-2, len_batch=10, n_epochs=1000)

                images = rbm.generer_image_RBM(nb_images=10, nb_iter=200)
                filename = f"results/rbm/hyperparameters/hidden_{q}_chars_{'-'.join(caractere)}.png"
                save_images(images,size_img,filename)


if __name__ == "__main__":

    RBM.test_rbm_hyperparameters()
