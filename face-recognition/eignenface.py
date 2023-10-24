from time import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Dataset
from sklearn.datasets import fetch_lfw_people
# Models
from sklearn.decomposition import PCA
import sklearn.manifold as manifold

from src.utils import image_utils as iu
from src.utils import init
# Use [skorch](https://github.com/skorch-dev/skorch). Install:
# `conda install -c conda-forge skorch`
device = init.init_device()
# download the data
lfw_people = init.init_people()
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = init.init_sample(lfw_people)

# split data into training and test set
X_train, X_test, y_train, y_test = init.init_data(lfw_people, n_samples)

target_names = lfw_people.target_names
n_classes = target_names.shape[0]

n_components = 150
print('Extracting the top %d eigenfaces from %d faces' % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print(f'done in {time() - t0:.3f}s')
eigenfaces = pca.components_.reshape((n_components, h, w))
print('Explained variance', pca.explained_variance_ratio_[:2])

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X_train)

print('Projecting the input data on the eigenfaces orthonormal basis')
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
df = pd.DataFrame(dict(lab=y_train,
        PC1=X_train_pca[:, 0],
        PC2=X_train_pca[:, 1],
        TSNE1=X_tsne[:, 0],
        TSNE2=X_tsne[:, 1])
    )
sns.relplot(x='PC1', y='PC2', hue='lab', data=df)
sns.relplot(x='TSNE1', y='TSNE2', hue='lab', data=df)
plt.show()

eigenface_titles = ['eigenface %d' % i for i in range(eigenfaces.shape[0])]
iu.plot_gallery(eigenfaces, eigenface_titles, h, w)