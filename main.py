import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loading_data import process_images_MNIST, load_idx3_ubyte
from principal_DNN_MNIST import DNN
from utils import one_hot_encoding

import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

"""
Analysis
"""

nb_classes = 10
len_batch = 64
learning_rate = 1e-2
nb_iter_RBM = 100
nb_iter_backprop = 200

train_images, train_size_img = process_images_MNIST('data/train-images-idx3-ubyte')
train_labels = load_idx3_ubyte('data/train-labels-idx1-ubyte')
encoded_train_labels = one_hot_encoding(train_labels, nb_classes)

test_images, test_size_img = process_images_MNIST('data/t10k-images-idx3-ubyte')
test_labels = load_idx3_ubyte('data/t10k-labels-idx1-ubyte')

nb_features = train_size_img[0]*train_size_img[1]

# dataframe for analysis
to_study = pd.DataFrame(columns=['Layers', 'Neurons', 'Train data', 'Error Train', 'Error Test'])

nb_layers_dbn = np.arange(1,6)
tmp = pd.DataFrame(data={'Layers': nb_layers_dbn, 'Neurons': [200]*len(nb_layers_dbn), 'Train data': [len(train_images)]*len(nb_layers_dbn)})
to_study = pd.concat([to_study, tmp], axis=0, ignore_index=True)

nb_neurons_dbn = np.arange(1,8)*100
tmp = pd.DataFrame(data={'Layers': [2]*len(nb_neurons_dbn), 'Neurons': nb_neurons_dbn, 'Train data': [len(train_images)]*len(nb_neurons_dbn)})
to_study = pd.concat([to_study, tmp], axis=0, ignore_index=True)

nb_train_data = [1000, 3000, 7000, 10000, 30000, 60000]
tmp = pd.DataFrame(data={'Layers': [2]*len(nb_train_data), 'Neurons': [200]*len(nb_train_data), 'Train data': nb_train_data})
to_study = pd.concat([to_study, tmp], axis=0, ignore_index=True)

to_study.drop_duplicates(subset=["Layers", "Neurons", "Train data"], ignore_index=True, inplace=True)
to_study.sort_values(["Train data", "Layers", "Neurons"], ignore_index=True, inplace = True)

to_study["Pre-trained"] = False

to_study = pd.concat([to_study, to_study], ignore_index=True)
to_study.loc[len(to_study)//2:, "Pre-trained"] = True

def train_dnn(network_size, train_images, encoded_train_labels, pretrain=True):
    dnn = DNN(network_size=network_size)
    if pretrain:
        dnn.pretrain_DNN(train_images, learning_rate=learning_rate, len_batch=len_batch, n_epochs=nb_iter_RBM, verbose=0)
    dnn.retropropagation(train_images, encoded_train_labels, learning_rate=learning_rate, len_batch=len_batch, n_epochs=nb_iter_backprop, verbose=0)
    return dnn

def run_analysis():

    for i, row in to_study.iterrows():
        nb_layers_dbn = row["Layers"]
        nb_neurons_dbn = row["Neurons"]
        nb_data = row["Train data"]
        pretrain = row["Pre-trained"]

        network_size = [nb_features] + [nb_neurons_dbn for i in range(nb_layers_dbn)] + [nb_classes]

        trained_dnn = train_dnn(network_size, train_images[:nb_data], encoded_train_labels[:nb_data], pretrain=pretrain)

        error_rate_train = trained_dnn.test_DNN(train_images, train_labels)
        error_rate_test = trained_dnn.test_DNN(test_images, test_labels)

        to_study.iloc[i, 3] = error_rate_train * 100
        to_study.iloc[i, 4] = error_rate_test * 100

        if i % 5 == 0:
            print(f"Run - {i}/{len(to_study) - 1}")

run_analysis()
to_study.to_csv("Analysis.csv", index=False)
######################################################################################################################
pretrained = to_study.query("`Pre-trained` == True and Neurons == 200 and `Train data` == 60000")
without_pretrained = to_study.query("`Pre-trained` == False and Neurons == 200 and `Train data` == 60000")

nb_layers_dbn = pretrained["Layers"].values

e_pretrained_dnn_train = pretrained["Error Train"].values
e_pretrained_dnn_test = pretrained["Error Test"].values
e_dnn_train = without_pretrained["Error Train"].values
e_dnn_test = without_pretrained["Error Test"].values

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(nb_layers_dbn, e_pretrained_dnn_train, label='Pre-trained DNN', marker='o', color='blue')
axs[0].set_xlabel('Number of Hidden Layers in DBN')
axs[0].set_ylabel('Pre-Trained - Error Rate (%)', color='blue')
axs[0].tick_params(axis='y', labelcolor='blue')
axs[0].set_title('Train')
axs[0].grid(True)

axs_twin_train = axs[0].twinx()
axs_twin_train.plot(nb_layers_dbn, e_dnn_train, label='Standard DNN', marker='o', color='orange')
axs_twin_train.set_ylabel('Standard - Error Rate (%)', color='orange')
axs_twin_train.tick_params(axis='y', labelcolor='orange')

axs[1].plot(nb_layers_dbn, e_pretrained_dnn_test, label='Pre-trained DNN', marker='o', color='blue')
axs[1].set_xlabel('Number of Hidden Layers in DBN')
axs[1].set_ylabel('Pre-Trained - Error Rate (%)', color='blue')
axs[1].tick_params(axis='y', labelcolor='blue')
axs[1].set_title('Test')
axs[1].grid(True)

axs_twin_test = axs[1].twinx()
axs_twin_test.plot(nb_layers_dbn, e_dnn_test, label='Standard DNN', marker='o', color='orange')
axs_twin_test.set_ylabel('Standard - Error Rate (%)', color='orange')
axs_twin_test.tick_params(axis='y', labelcolor='orange')

fig.suptitle('Number of Layers in DBN', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.savefig(f'img/Nb_Layers_{int(learning_rate*100)}_{len_batch}.png')
# plt.show()
######################################################################################################################
pretrained = to_study.query("`Pre-trained` == True and Layers == 2 and `Train data` == 60000")
without_pretrained = to_study.query("`Pre-trained` == False and Layers == 2 and `Train data` == 60000")

nb_neurons_dbn = pretrained["Neurons"].values

e_pretrained_dnn_train = pretrained["Error Train"].values
e_pretrained_dnn_test = pretrained["Error Test"].values
e_dnn_train = without_pretrained["Error Train"].values
e_dnn_test = without_pretrained["Error Test"].values

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(nb_neurons_dbn, e_pretrained_dnn_train, label='Pre-trained DNN', marker='o', color='blue')
axs[0].set_xlabel('Number of Neurons in DBN')
axs[0].set_ylabel('Pre-Trained - Error Rate (%)', color='blue')
axs[0].tick_params(axis='y', labelcolor='blue')
axs[0].set_title('Train')
axs[0].grid(True)

axs_twin_train = axs[0].twinx()
axs_twin_train.plot(nb_neurons_dbn, e_dnn_train, label='Standard DNN', marker='o', color='orange')
axs_twin_train.set_ylabel('Standard - Error Rate (%)', color='orange')
axs_twin_train.tick_params(axis='y', labelcolor='orange')

axs[1].plot(nb_neurons_dbn, e_pretrained_dnn_test, label='Pre-trained DNN', marker='o', color='blue')
axs[1].set_xlabel('Number of Neurons in DBN')
axs[1].set_ylabel('Pre-Trained - Error Rate (%)', color='blue')
axs[1].tick_params(axis='y', labelcolor='blue')
axs[1].set_title('Test')
axs[1].grid(True)

axs_twin_test = axs[1].twinx()
axs_twin_test.plot(nb_neurons_dbn, e_dnn_test, label='Standard DNN', marker='o', color='orange')
axs_twin_test.set_ylabel('Standard - Error Rate (%)', color='orange')
axs_twin_test.tick_params(axis='y', labelcolor='orange')

fig.suptitle('Number of Neurons in DBN', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.savefig(f'img/Nb_Neurons_{int(learning_rate*100)}_{len_batch}.png')
# plt.show()
######################################################################################################################
pretrained = to_study.query("`Pre-trained` == True and Layers == 2 and Neurons == 200")
without_pretrained = to_study.query("`Pre-trained` == False and Layers == 2 and Neurons == 200")

nb_data = pretrained["Train data"].values

e_pretrained_dnn_train = pretrained["Error Train"].values
e_pretrained_dnn_test = pretrained["Error Test"].values
e_dnn_train = without_pretrained["Error Train"].values
e_dnn_test = without_pretrained["Error Test"].values

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(nb_data, e_pretrained_dnn_train, label='Pre-trained DNN', marker='o', color='blue')
axs[0].set_xlabel('Number of Data')
axs[0].set_ylabel('Pre-Trained - Error Rate (%)', color='blue')
axs[0].tick_params(axis='y', labelcolor='blue')
axs[0].set_title('Train')
axs[0].grid(True)

axs_twin_train = axs[0].twinx()
axs_twin_train.plot(nb_data, e_dnn_train, label='Standard DNN', marker='o', color='orange')
axs_twin_train.set_ylabel('Standard - Error Rate (%)', color='orange')
axs_twin_train.tick_params(axis='y', labelcolor='orange')

axs[1].plot(nb_data, e_pretrained_dnn_test, label='Pre-trained DNN', marker='o', color='blue')
axs[1].set_xlabel('Number of Data')
axs[1].set_ylabel('Pre-Trained - Error Rate (%)', color='blue')
axs[1].tick_params(axis='y', labelcolor='blue')
axs[1].set_title('Test')
axs[1].grid(True)

axs_twin_test = axs[1].twinx()
axs_twin_test.plot(nb_data, e_dnn_test, label='Standard DNN', marker='o', color='orange')
axs_twin_test.set_ylabel('Standard - Error Rate (%)', color='orange')
axs_twin_test.tick_params(axis='y', labelcolor='orange')

fig.suptitle('Number of Data', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.savefig(f'img/Nb_Data_{int(learning_rate*100)}_{len_batch}.png')
#plt.show()
