from time import time
from sklearn.metrics import classification_report
# Preprocesing
from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# Pytorch Models
import torch
# import torchvision
from skorch import NeuralNetClassifier
import skorch
# import utils
from src.utils import init
from src.models.cnn import Cnn
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

torch.manual_seed(0)
cnn = NeuralNetClassifier(
        Cnn,
        max_epochs=100,
        lr=0.001,
        optimizer=torch.optim.Adam,
        device=device,
        train_split=skorch.dataset.ValidSplit(cv=5, stratified=True),
        verbose=0,
    )
scaler = preprocessing.MinMaxScaler()
X_train_s = scaler.fit_transform(X_train).reshape(-1, 1, h, w)
X_test_s = scaler.transform(X_test).reshape(-1, 1, h, w)
t0 = time()
cnn.fit(X_train_s, y_train)
print(f'done in {time() - t0:.3f}s')
y_pred = cnn.predict(X_test_s)
print(classification_report(y_test, y_pred, target_names=target_names))