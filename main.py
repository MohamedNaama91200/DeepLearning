import numpy as np
import matplotlib.pyplot as plt

from loading_data import process_images_MNIST, load_idx3_ubyte
from principal_DNN_MNIST import DNN
from utils import one_hot_encoding

"""
Main Code for Deep Neural Networks 
"""

nb_classes = 10

train_images, train_size_img = process_images_MNIST('data/train-images-idx3-ubyte')
train_labels = load_idx3_ubyte('data/train-labels-idx1-ubyte')
encoded_train_labels = one_hot_encoding(train_labels, nb_classes)

test_images, test_size_img = process_images_MNIST('data/t10k-images-idx3-ubyte')
test_labels = load_idx3_ubyte('data/t10k-labels-idx1-ubyte')

nb_features = train_size_img[0]*train_size_img[1]

def train_and_test_dnn(network_size, train_images, encoded_train_labels, test_images, test_labels, pretrain=True):
    dnn = DNN(network_size=network_size)
    if pretrain:
        dnn.pretrain_DNN(train_images, learning_rate=1e-2, len_batch=128, n_epochs=100, verbose=0)
    dnn.retropropagation(train_images, encoded_train_labels, learning_rate=1e-2, len_batch=128, n_epochs=200, verbose=0)
    return dnn.test_DNN(test_images, test_labels)


######################################################################################################################
nb_layers_dbn = np.arange(1,6)

networks_size = [[nb_features] + [200]*i + [nb_classes] for i in nb_layers_dbn]

e_pretrained_dnn = []
e_dnn = []

for s in networks_size:
    error_rate = train_and_test_dnn(s, train_images, encoded_train_labels, test_images, test_labels)
    e_pretrained_dnn.append(error_rate)

    error_rate = train_and_test_dnn(s, train_images, encoded_train_labels, test_images, test_labels, pretrain=False)
    e_dnn.append(error_rate)

fig, axs = plt.subplots(1, 2, figsize=(9, 4))

axs[0].plot(nb_layers_dbn, e_pretrained_dnn, label='Pre-trained DNN', marker='o')
axs[0].set_title('Pre-trained DNN')
axs[0].set_xlabel('Number of Hidden layers in DBN')
axs[0].set_ylabel('Error Rate')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(nb_layers_dbn, e_dnn, label='Standard DNN', marker='o', color='orange')
axs[1].set_title('Standard DNN')
axs[1].set_xlabel('Number of Hidden layers in DBN')
axs[1].set_ylabel('Error Rate')
axs[1].grid(True)
axs[1].legend()

fig.suptitle('Error Rate vs. Number of Layers in DBN', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.show()

######################################################################################################################
nb_neurons_dbn = np.arange(1,8)*100

networks_size = [[nb_features] + [n,n] + [nb_classes] for n in nb_neurons_dbn]

e_pretrained_dnn = []
e_dnn = []

for s in networks_size:
    error_rate = train_and_test_dnn(s, train_images, encoded_train_labels, test_images, test_labels)
    e_pretrained_dnn.append(error_rate)

    error_rate = train_and_test_dnn(s, train_images, encoded_train_labels, test_images, test_labels, pretrain=False)
    e_dnn.append(error_rate)

fig, axs = plt.subplots(1, 2, figsize=(9, 4))

axs[0].plot(nb_neurons_dbn, e_pretrained_dnn, label='Pre-trained DNN', marker='o')
axs[0].set_title('Pre-trained DNN')
axs[0].set_xlabel('Number of Neurons')
axs[0].set_ylabel('Error Rate')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(nb_neurons_dbn, e_dnn, label='Standard DNN', marker='o', color='orange')
axs[1].set_title('Standard DNN')
axs[1].set_xlabel('Number of Neurons')
axs[1].set_ylabel('Error Rate')
axs[1].grid(True)
axs[1].legend()

fig.suptitle('Error Rate vs. Number of Neurons of Hidden Layers in DBN', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.show()

######################################################################################################################
nb_train_data = [1000, 3000, 7000, 10000, 30000, 60000]

network_size = [nb_features] + [200,200] + [nb_classes]

e_pretrained_dnn = []
e_dnn = []

for n in nb_train_data:
    error_rate = train_and_test_dnn(network_size, train_images[:n], encoded_train_labels[:n], test_images, test_labels)
    e_pretrained_dnn.append(error_rate)

    error_rate = train_and_test_dnn(network_size, train_images, encoded_train_labels, test_images, test_labels, pretrain=False)
    e_dnn.append(error_rate)

fig, axs = plt.subplots(1, 2, figsize=(9, 4))

axs[0].plot(nb_train_data, e_pretrained_dnn, label='Pre-trained DNN', marker='o')
axs[0].set_title('Pre-trained DNN')
axs[0].set_xlabel('Number of Training Data')
axs[0].set_ylabel('Error Rate')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(nb_train_data, e_dnn, label='Standard DNN', marker='o', color='orange')
axs[1].set_title('Standard DNN')
axs[1].set_xlabel('Number of Training Data')
axs[1].set_ylabel('Error Rate')
axs[1].grid(True)
axs[1].legend()

fig.suptitle('Error Rate vs. Number of Neurons of Hidden Layers in DBN', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.show()

