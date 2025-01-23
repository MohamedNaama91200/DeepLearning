import copy

import numpy as np

from loading_data import process_images_MNIST, load_idx3_ubyte
from principal_DBN_alpha import DBN
from utils import calcul_softmax, one_hot_encoding

"""
DNN Class Structure
"""


class DNN:
    def __init__(self, network_size):
        """
        Args:
            network_size: size of the network including the layer for classification
        """
        self.dbn = DBN(network_size[:-1])
        nb_classes = network_size[-1]
        self.W_l = np.random.randn(network_size[-2], nb_classes) * np.sqrt(0.01)  # Weights for classification layer
        self.b_l = np.zeros(nb_classes)  # Bias for classification layer

    def pretrain_DNN(self, X, learning_rate, len_batch, n_epochs, verbose=1):
        self.dbn.train_DBN(X, learning_rate, len_batch, n_epochs, verbose)

    def entree_sortie_reseau(self, X):
        """
        Store and return inputs + the outputs of each layer.
        """

        outputs = [X]

        h = copy.deepcopy(X)
        for rbm in self.dbn.rbms:
            h = rbm.entree_sortie_RBM(h)
            outputs.append(h)

        y_hat = calcul_softmax(h @ self.W_l + self.b_l)
        outputs.append(y_hat)

        return outputs

    def retropropagation(self, X, labels, learning_rate, len_batch, n_epochs, verbose=1):
        """
        Perform backpropagation to fine-tune the DBN using labels.
        Args:
            labels (np.ndarray): one-hot encoded labels.
        """

        n_samples = X.shape[0]

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = labels[indices]

            for ith_batch in range(0, n_samples, len_batch):

                X_batch = X_shuffled[ith_batch:ith_batch + len_batch]
                y_batch = y_shuffled[ith_batch:ith_batch + len_batch]

                m = X_batch.shape[0]

                # Forward pass
                outputs = self.entree_sortie_reseau(X_batch)
                y_hat = outputs[-1]

                c = y_hat - y_batch

                # Backward pass
                grad_w = outputs[-2].T @ c / m
                grad_b = np.mean(c, axis=0)

                weight = copy.deepcopy(self.W_l)

                self.W_l -= learning_rate * grad_w
                self.b_l -= learning_rate * grad_b

                for l in range(len(outputs) - 2, 0, -1):
                    x = outputs[l]
                    c = (c @ weight.T) * (x * (1 - x))

                    x_prev = outputs[l - 1]
                    grad_w = x_prev.T @ c / m
                    grad_b = np.mean(c, axis=0)

                    rbm = self.dbn.rbms[l - 1]
                    weight = copy.deepcopy(rbm.W)

                    # Update RBM weights and biases
                    rbm.W -= learning_rate * grad_w
                    rbm.b -= learning_rate * grad_b

            # Reconstruction of the Cross-entropy loss
            outputs = self.entree_sortie_reseau(X_shuffled)
            y_hat = outputs[-1]

            # Cross-entropy loss
            loss = -np.sum(y_shuffled * np.log(y_hat)) / n_samples

            if epoch % 10 == 0 and verbose:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")

    def test_DNN(self, X, labels):
        """
        Compute the error rate.

        Args:
            labels (np.ndarray).
        """

        # Retrieve estimated label
        outputs = self.entree_sortie_reseau(X)
        y_hat = outputs[-1]
        label_estimated = np.argmax(y_hat, axis=1)

        errors = [0 if x == y else 1 for (x, y) in zip(labels, label_estimated)]

        return sum(errors) / len(errors)

if __name__ == "__main__":
    nb_classes = 10

    train_images, train_size_img = process_images_MNIST('data/train-images-idx3-ubyte')
    train_labels = load_idx3_ubyte('data/train-labels-idx1-ubyte')
    encoded_train_labels = one_hot_encoding(train_labels, nb_classes)

    test_images, test_size_img = process_images_MNIST('data/t10k-images-idx3-ubyte')
    test_labels = load_idx3_ubyte('data/t10k-labels-idx1-ubyte')

    nb_features = train_size_img[0] * train_size_img[1]

    dnn_pretrained = DNN(network_size=[nb_features, 200, 200, nb_classes])
    print(
        "----------------------------------------------------- Pre-training -----------------------------------------------------")
    dnn_pretrained.pretrain_DNN(train_images, learning_rate=1e-2, len_batch=32, n_epochs=10)
    print(
        "----------------------------------------------------- Back-Propragation -----------------------------------------------------")
    dnn_pretrained.retropropagation(train_images, encoded_train_labels, learning_rate=1e-2, len_batch=32, n_epochs=20)
    print(
        "----------------------------------------------------- Error Rate -----------------------------------------------------")
    error_rate = dnn_pretrained.test_DNN(test_images, test_labels)
    print(f"Error rate: {error_rate*100}%")























