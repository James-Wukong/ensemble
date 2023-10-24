from time import time
from sklearn.model_selection import train_test_split

# Dataset
from sklearn.datasets import fetch_lfw_people

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# Pytorch Models
import torch
import src.utils.image_utils as iu

# Use [skorch](https://github.com/skorch-dev/skorch). Install:
# `conda install -c conda-forge skorch`
def init_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def init_people():
    # download the data
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people

def init_sample(lfw_people):
    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    return n_samples, h, w

def init_data(lfw_people, n_samples):
    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]
    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    print('Total dataset size:')
    print('n_samples: %d' % n_samples)
    print('n_features: %d' % n_features)
    print('n_classes: %d' % n_classes)

    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    print({target_names[lab]: prop for lab, prop in iu.label_proportion(y_train).items()})

    return X_train, X_test, y_train, y_test