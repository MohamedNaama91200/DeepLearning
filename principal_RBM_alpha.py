import numpy as np
import os
from loading_data import lire_alpha_digit
from utils import sigmoid,save_images
import matplotlib.pyplot as plt

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

        loss_history = []

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
            loss_history.append(loss)

        return loss_history

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

    @staticmethod
    def test_rbm_learning_rate():
        os.makedirs("results/rbm/hyperparameters", exist_ok=True)

        # Plage de learning rates et tailles de batch
        learning_rates = np.linspace(0.01, 0.1, 10)
        q = 200  # Nombre fixé de neurones cachés
        caractere = ['A', 'B', 'C']  # Caractères fixes

        # Chargement des données
        X, size_img = lire_alpha_digit('data/binaryalphadigs.mat', caractere=caractere)
        p = size_img[0] * size_img[1]  # Taille des entrées

        # Configuration des graphiques
        plt.figure(figsize=(12, 7))
        cmap = plt.get_cmap('plasma')  # Palette de couleurs
        colors = cmap(np.linspace(0, 1, len(learning_rates)))

        color_idx = 0  # Indice pour les couleurs

        for lr in learning_rates:
            # Initialisation et entraînement du RBM
            rbm = RBM(p, q)
            print(f"Training with learning rate: {lr:.3f}, batch size: 10")
            loss_history = rbm.train_RBM(
                X,
                learning_rate=lr,
                len_batch=10,
                n_epochs=100  # Nombre d'epochs
            )

            # Tracé de la courbe de loss
            plt.plot(
                loss_history,
                color=colors[color_idx],  # Couleur unique pour chaque combinaison
                linewidth=1.5,
                label=f"η={lr:.3f}, batch=10"
            )
            color_idx += 1

        # Personnalisation du graphique
        plt.title("Loss vs Epochs for each L.R and Batch Size=10", pad=20, fontsize=14)
        plt.xlabel("Epochs", labelpad=15, fontsize=12)
        plt.ylabel("Loss", labelpad=15, fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Hyperparameters")  # Légende externalisée
        plt.grid(alpha=0.3)  # Grille discrète pour plus de lisibilité
        plt.tight_layout()  # Ajustement automatique des marges

        # Sauvegarde du graphique
        plot_filename = f"results/rbm/hyperparameters/lr_batch_comparison_q{q}.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Plot saved as {plot_filename}")


    @staticmethod
    def test_rbm_lenbatch():
        os.makedirs("results/rbm/hyperparameters", exist_ok=True)

        # Plage de tailles de batch
        len_batches = [5, 10, 20, 50]
        q = 200  # Nombre fixé de neurones cachés
        caractere = ['A', 'B', 'C']  # Caractères fixes

        # Chargement des données
        X, size_img = lire_alpha_digit('data/binaryalphadigs.mat', caractere=caractere)
        p = size_img[0] * size_img[1]  # Taille des entrées

        # Configuration des graphiques
        plt.figure(figsize=(12, 7))
        cmap = plt.get_cmap('plasma')  # Palette de couleurs
        colors = cmap(np.linspace(0, 1, len(len_batches)))

        color_idx = 0  # Indice pour les couleurs

        for batch in len_batches:
            # Initialisation et entraînement du RBM
            rbm = RBM(p, q)
            print(f"Training with batch size: {batch}")
            loss_history = rbm.train_RBM(
                X,
                learning_rate=0.01,
                len_batch=batch,
                n_epochs=1000  # Nombre d'epochs
            )

            # Tracé de la courbe de loss
            plt.plot(
                loss_history,
                color=colors[color_idx],  # Couleur unique pour chaque combinaison
                linewidth=1.5,
                label=f"batch={batch}"
            )
            color_idx += 1


        # Personnalisation du graphique
        plt.title("Loss vs Epochs for each Batch Size", pad=20, fontsize=14)
        plt.xlabel("Epochs", labelpad=15, fontsize=12)
        plt.ylabel("Loss", labelpad=15, fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Hyperparameters")  # Légende externalisée
        plt.grid(alpha=0.3)  # Grille discrète pour plus de lisibilité
        plt.tight_layout()  # Ajustement automatique des marges

        # Sauvegarde du graphique
        plot_filename = f"results/rbm/hyperparameters/lr_batch_comparison_{q}.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Plot saved as {plot_filename}")

if __name__ == "__main__":

    #X,size_img = lire_alpha_digit('data/binaryalphadigs.mat', caractere=['A', 'B','C'])
    #print(size_img)
    #print(len(X))
    #save_images(X, size_img, filename="results/binary_alpha_digit.png")
    #RBM.test_rbm_hyperparameters()
    #RBM.test_rbm_learning_rate()
    RBM.test_rbm_lenbatch()