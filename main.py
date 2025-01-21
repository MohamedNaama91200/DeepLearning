from loading_data import lire_alpha_digit
from principal_RBM_alpha import RBM
from principal_DBN_alpha import DBN
from utils import plot_images

"""
Main Code for Deep Neural Networks 
"""
X, size_img = lire_alpha_digit(caractere=['A'])

"""
#Random images from the BinAlphaDigit dataset
for _ in range(5):
    i = np.random.choice(X.shape[0])
    plot_images([X[i]],database='BinaryAlphaDigit')
"""

"""
Training RBM 
"""

"""
p,q = size_img[0]*size_img[1], 100
rbm = RBM(p,q) #Instance of RBM
rbm.train_RBM(X, learning_rate=10**(-2), len_batch=10, n_epochs=1000, verbose=1)

generated_images_rbm = rbm.generer_image_RBM(X=X,nb_images=10, nb_iter=200, size_img=size_img)
#plot_images(generated_images, database='BinaryAlphaDigit')
"""

"""
Training DBN
"""


dbn_size = [320, 200, 100]
dbn = DBN(dbn_size) #Instance of RBM
dbn.train_DBN(X, learning_rate=10**(-2), len_batch=10, n_epochs=1000)

generated_images = dbn.generer_image_DBN(X=X,nb_images=10, nb_iter=200, size_img=size_img)
plot_images(generated_images, database='BinaryAlphaDigit')

