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


def lire_alpha_digit(file_path='data/binaryalphadigs.mat', caractere=['0']):
    assert caractere, "List not empty"

    elements = [str(i) for i in range(10)] + list(string.ascii_uppercase)

    index_caractere_list = [elements.index(c) for c in caractere if c in elements]
    if not index_caractere_list:
        raise ValueError("One or many caracters are not recognized.")

    data = loadmat(file_path)

    size_img = data['dat'][0][0].shape
    nb_pixel = size_img[0] * size_img[1]

    X = data['dat'][np.array(index_caractere_list)]
    X = np.concatenate(X)
    X = np.concatenate(X).reshape((X.shape[0], nb_pixel))

    return X, size_img

def process_images_MNIST(file_path='data/train-images-idx3-ubyte'):
    """
    Loads MNIST image data and reshapes them into a 2D array where each row corresponds to an image,
    and binarizes the pixel values by normalizing and rounding.
    """
    images = load_idx3_ubyte(file_path)

    size_img = images[0].shape
    images = images.reshape((images.shape[0], size_img[0] * size_img[1]))

    images = np.round(images / 255)  # binary MNIST

    return images, size_img

if __name__ == "__main__":
    images = load_idx3_ubyte('data/t10k-images.idx3-ubyte')
    labels = load_idx3_ubyte('data/t10k-labels-idx1-ubyte')

    mat_data,size_img = lire_alpha_digit('data/binaryalphadigs.mat',caractere=['A','Z'])

    print("Images shape:", images[0].shape)
    print("Labels shape:", labels[0])
    #print("Mat data length:", size_img)
    #print(mat_data)

