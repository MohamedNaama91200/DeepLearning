import numpy as np

import utils
from principal_DBN_alpha import DBN
"""
DNN Class Structure
"""


class DNN:
    def __init__(self,network_size,n_classes):
        """
        :param network_size: taille du réseau : la dernière est la couche de classif.
        :param n_classes: Nombre de classes pour la couche de classification.
        """
        self.dbn_without_classif_layer = DBN(network_size[:-1])
        self.n_classes = n_classes
        self.weights_classif = np.random.randn(network_size[-1], n_classes) * 0.01  # Poids couche classif (loi normale)
        self.bias_classif = np.zeros(n_classes)  # Biais de la couche de classification


    def pretrain_DNN(self,X,learning_rate, len_batch, n_epochs) :

        self.dbn_without_classif_layer.train_DBN(X,learning_rate, len_batch, n_epochs)


    def calcul_softmax(self,X):
        return utils.sigmoid(X)

    def entree_sortie_reseau(self,X):

        sortie_couche = [X]
        h = X
        for rbm in self.dbn_without_classif_layer.rbms :
            h = rbm.entree_sortie_RBM(h)
            sortie_couche.append(h)

        probs_sortie = self.calcul_softmax(h @ self.weights_classif + self.bias_classif)

        sortie_couche.append(probs_sortie)

        return sortie_couche


    def retropropagation(self):

        return self


if __name__ == "__main__":

    from loading_data import load_idx3_ubyte
    from utils import plot_images

    images = load_idx3_ubyte('data/train-images-idx3-ubyte')
    labels = load_idx3_ubyte('data/train-labels-idx1-ubyte')
    dnn = DNN(network_size=[784, 200, 100],n_classes=28)
    images = np.reshape(images,(60000,784))
    images = images / 255 #normalisation en niveau de noir ou blanc



    dnn.pretrain_DNN(images,learning_rate=10**(-2), len_batch=10, n_epochs=2)
    generated_images = dnn.dbn_without_classif_layer.generer_image_DBN(X=images, nb_images=10, nb_iter=200, size_img=784)
    plot_images(generated_images,database='MNIST')




















