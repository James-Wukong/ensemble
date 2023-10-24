from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# Preprocesing
from sklearn import preprocessing
# Pytorch Models
import torch
# import torchvision
import torch.nn as nn
from skorch import NeuralNetClassifier
import skorch
# import utils
from src.utils import init
from src.models.cnn import Resnet18
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
resnet = NeuralNetClassifier(
        Resnet18,
        # `CrossEntropyLoss` combines `LogSoftmax and `NLLLoss`
        criterion=nn.CrossEntropyLoss,
        max_epochs=50,
        batch_size=128, # default value
        optimizer=torch.optim.Adam,
        # optimizer=torch.optim.SGD,
        optimizer__lr=0.001,
        optimizer__betas=(0.9, 0.999),
        optimizer__eps=1e-4,
        optimizer__weight_decay=0.0001, # L2 regularization
        # Shuffle training data on each epoch
        # iterator_train__shuffle=True,
        train_split=skorch.dataset.ValidSplit(cv=5, stratified=True),
        device=device,
        verbose=0
    )
scaler = preprocessing.MinMaxScaler()
X_train_s = scaler.fit_transform(X_train).reshape(-1, 1, h, w)
X_test_s = scaler.transform(X_test).reshape(-1, 1, h, w)
t0 = time()
resnet.fit(X_train_s, y_train)
print(f'done in {time() - t0:.3f}s')

# Continue training a model (warm re-start):
# resnet.partial_fit(X_train_s, y_train)
y_pred = resnet.predict(X_test_s)
print(classification_report(y_test, y_pred, target_names=target_names))
epochs = np.arange(len(resnet.history[:, 'train_loss'])) + 1
plt.plot(epochs, resnet.history[:, 'train_loss'], '-b', label='train_loss')
plt.plot(epochs, resnet.history[:, 'valid_loss'], '-r', label='valid_loss')
plt.plot(epochs, resnet.history[:, 'valid_acc'], '--r', label='valid_acc')
plt.legend()
plt.show()