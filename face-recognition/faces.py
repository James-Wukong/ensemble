from time import time
import numpy as np
# import utils
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

# plot mean faces and 4 samples fo each individual
single_faces = [X_train[y_train == lab][:5] for lab in np.unique(y_train)]
single_faces = np.vstack(single_faces).reshape((5 * n_classes, h, w))
mean_faces = [X_train[y_train == lab].mean(axis=0) for lab in np.unique(y_train)]
mean_faces = np.vstack(mean_faces).reshape((n_classes, h, w))
single_faces[::5, :, :] = mean_faces
titles = [n for name in target_names for n in [name] * 5]
iu.plot_gallery(single_faces, titles, h, w, n_row=n_classes, n_col=5)