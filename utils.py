import numpy as np
import matplotlib.pyplot as plt

"""
Useful functions
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_images(X, size_img):
    num_images = len(X)

    # Calculate the number of rows and columns for the subplots
    cols = np.ceil(np.sqrt(num_images))
    rows = np.ceil(num_images / cols)

    fig, axes = plt.subplots(int(rows), int(cols), figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, image in enumerate(X):
        image = image.reshape(size_img)
        axes[i].imshow(image, cmap='gray')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def save_images(X, size_img, filename):
    num_images = len(X)

    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, image in enumerate(X):
        image = image.reshape(size_img)
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Ajustement de la mise en page et sauvegarde
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def one_hot_encoding(labels, nb_classes):
    """
    Converts a label vector into a one-hot encoded matrix.
    """
    one_hot = np.zeros((len(labels), nb_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def calcul_softmax(x):
    """
    Args:
        x (np.ndarray)
    """
    assert isinstance(x, np.ndarray), "Please use array."

    if x.ndim == 1:
        return np.exp(x) / np.sum(np.exp(x))
    elif x.ndim == 2:
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    else:
        raise ValueError("1 or 2 dimensional array.")