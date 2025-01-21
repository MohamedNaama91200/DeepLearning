import numpy as np
from scipy.io.matlab import loadmat
import string

"""
Binary AlphaDigits and MNIST DB
"""

def load_idx3_ubyte(file_path='data/t10k-images.idx3-ubyte'):

    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_items = int.from_bytes(f.read(4), 'big')

        if magic_number == 2051:  # Fichier d'images
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_items, num_rows, num_cols)
        elif magic_number == 2049:  # Fichier d'étiquettes
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError("Format de fichier .idx3-ubyte non reconnu")

    data = np.copy(data)

    return data


def lire_alpha_digit(file_path='data/binaryalphadigs.mat',caractere=['0']):

    elements = [str(i) for i in range(10)] + list(string.ascii_uppercase)
    try :
        mapping = {elements[k]: k for k in range(36)}
        index_caractere_list = [mapping[c] for c in caractere]
    except :
        raise ValueError("Caractère non reconnu")

    data = loadmat(file_path)
    size_img = data['dat'][0][0].shape
    X=data['dat'][index_caractere_list[0]]
    for i in range(1,len(index_caractere_list)) :
        X_bis=data['dat'][index_caractere_list[i]]
        X=np.concatenate((X,X_bis),axis=0)

    X=np.concatenate(X).reshape((X.shape[0],320))

    return X,size_img





#For testing purpose
if __name__ == "__main__":
    images = load_idx3_ubyte('data/t10k-images.idx3-ubyte')
    labels = load_idx3_ubyte('data/t10k-labels-idx1-ubyte')

    mat_data,size_img = lire_alpha_digit('data/binaryalphadigs.mat',caractere=['A','Z'])

    print("Images shape:", images[0].shape)
    print("Labels shape:", labels[0])
    #print("Mat data length:", size_img)
    #print(mat_data)

