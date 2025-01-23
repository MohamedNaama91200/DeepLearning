import copy

import numpy as np
import utils
from principal_DBN_alpha import DBN

"""
DNN Class Structure
"""


class DNN:
    def __init__(self,network_layer,n_classes):
        """
        :param network_layer: taille du réseau : la dernière est la couche de classif.
        :param n_classes: Nombre de classes pour la couche de classification.
        """
        self.length_network = len(network_layer)
        self.dbn_without_classif_layer = DBN(network_layer[:-1])
        self.n_classes = n_classes
        self.W = {i+1 : rbm.W for i,rbm in enumerate(self.dbn_without_classif_layer.rbms)}
        self.b = {i+1: rbm.b for i,rbm in enumerate(self.dbn_without_classif_layer.rbms)}
        self.W[self.length_network-1] = np.random.randn(network_layer[-2], n_classes) * np.sqrt(0.01) # Poids couche classif (loi normale)
        self.b[self.length_network-1] = np.zeros(n_classes)  # Biais de la couche de classification


    def pretrain_DNN(self,X,learning_rate, len_batch, n_epochs) :
        X_input = copy.deepcopy(X)
        self.dbn_without_classif_layer.train_DBN(X_input,learning_rate, len_batch, n_epochs)

    def entree_sortie_reseau(self,X):

        X_input = copy.deepcopy(X)
        sortie_couche = [X_input]
        h = X_input
        #parcours des couches cachées
        for l in range(1, self.length_network-1):
            h = utils.sigmoid(h @ self.W[l] + self.b[l])
            sortie_couche.append(h)

        probs_sortie = utils.calcul_softmax(h @ self.W[self.length_network-1] + self.b[self.length_network-1])

        sortie_couche.append(probs_sortie)

        return sortie_couche


    def retropropagation(self, X, y, learning_rate, len_batch, n_epochs):

        n_samples = X.shape[0]
        history_loss = []

        for epoch in range(n_epochs):
            # Mélanger les données à chaque epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y.iloc[indices]

            for i in range(0, n_samples, len_batch):
                X_batch = X_shuffled[i:i + len_batch]
                y_batch = y_shuffled.iloc[i:i + len_batch]
                #Forward
                sortie_couche = self.entree_sortie_reseau(X_batch)
                y_pred = sortie_couche[-1]

                # Calcul de l'erreur (entropie croisée)
                loss = utils.calcul_entropie_croisee(y_pred, y_batch)
                history_loss.append(loss)

                # Backward
                # Gradient de la couche de classification (cf demo cours Petetin par Chain Rule)
                m = y_batch.shape[0]
                #d_logits = y_pred
                d_logits = y_pred - y_batch.to_numpy()  #C'est Dloss/Dzj avec zj les "logits",i.e les sorties avant softmax
                d_logits /= m

                h = sortie_couche[-2]
                d_weights_classif = h.T @ d_logits
                d_bias_classif = np.sum(d_logits, axis=0)
                self.W[self.length_network-1] -= learning_rate * d_weights_classif
                self.b[self.length_network-1] -= learning_rate * d_bias_classif

                # Rétropropagation à travers les couches cachées
                d_h = d_logits @ self.W[self.length_network-1].T
                for l in range(self.length_network-2,0,-1) :

                    h_prev = sortie_couche[l-1]
                    h_current = sortie_couche[l]
                    d_h *= h_current*(1-h_current) # dérivée de la sigmoide
                    d_weights = h_prev.T @ d_h #formule derivation de d_weights cf cours
                    d_bias = np.sum(d_h, axis=0)

                    # Mise à jour des poids et biais
                    self.W[l] -= learning_rate * d_weights
                    self.b[l] -= learning_rate * d_bias

                    # Propagation du gradient à la couche précédente
                    d_h = d_h @ self.W[l].T

            print(f"Epoch {epoch + 1}/{n_epochs}, Loss DNN: {loss}")

        return history_loss

    def test_DNN(self,X_test,y_test):

        sortie_couche = self.entree_sortie_reseau(X_test)
        y_pred = sortie_couche[-1]

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        error = np.mean(y_pred != y_true)

        return error


if __name__ == "__main__":

    from loading_data import load_idx3_ubyte

    #train data
    images_train = load_idx3_ubyte('data/train-images-idx3-ubyte')
    labels_train = load_idx3_ubyte('data/train-labels-idx1-ubyte')
    labels_train_encoded = utils.binarisation(labels_train)
    images_train = np.reshape(images_train,(60000,784))
    images_train = images_train / 255 #normalisation en niveau de noir ou blanc (RunTime Error overflow)

    #test data
    images_test = load_idx3_ubyte('data/t10k-images.idx3-ubyte')
    labels_test = load_idx3_ubyte('data/t10k-labels-idx1-ubyte')
    labels_test_encoded = utils.binarisation(labels_test)
    images_test = np.reshape(images_test,(10000,784))
    images_test = images_test / 255 #normalisation en niveau de noir ou blanc (RunTime Error overflow)



    #Training DNN
    dnn_without_pretraining = DNN(network_layer=[784, 128, 64, 10],n_classes=10)
    dnn_with_pretraining = DNN(network_layer=[784, 128, 64, 10],n_classes=10)

    dnn_with_pretraining.pretrain_DNN(images_train,learning_rate=10**(-2), len_batch=10, n_epochs=5)
    dnn_with_pretraining.retropropagation(X=images_train,y=labels_train_encoded,learning_rate=10**(-2), len_batch=10, n_epochs=5)
    dnn_without_pretraining.retropropagation(X=images_train,y=labels_train_encoded,learning_rate=10**(-2), len_batch=10, n_epochs=5)

    #Testing DNN on test set
    error = dnn_with_pretraining.test_DNN(X_test=images_test,y_test=labels_test_encoded)
    print(f"Error ratio with pre training {error}")

    error = dnn_without_pretraining.test_DNN(X_test=images_test,y_test=labels_test_encoded)
    print(f"Error ratio without pre-training {error}")

    generated_images = dnn_with_pretraining.dbn_without_classif_layer.generer_image_DBN(nb_images=10, nb_iter=500)
    utils.plot_images(generated_images,database='MNIST')





















