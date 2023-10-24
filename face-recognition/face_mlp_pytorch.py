from time import time
from sklearn.metrics import classification_report
# Preprocesing
from sklearn import preprocessing
# Pytorch Models
import torch
# import torchvision
from skorch import NeuralNetClassifier
from src.utils import init
from src.models.simple_mlp import SimpleMLPClassifierPytorch
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

mlp = NeuralNetClassifier( # Match the parameters with sklearn
        SimpleMLPClassifierPytorch,
        criterion=torch.nn.NLLLoss,
        max_epochs=100,
        batch_size=200,
        optimizer=torch.optim.Adam,
        # optimizer=torch.optim.SGD,
        optimizer__lr=0.001,
        optimizer__betas=(0.9, 0.999),
        optimizer__eps=1e-4,
        optimizer__weight_decay=0.0001, # L2 regularization
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        device=device,
        verbose=0
    )

scaler = preprocessing.MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

t0 = time()
mlp.fit(X_train_s, y_train)
print(f'done in {time() - t0:.3f}s')
y_pred = mlp.predict(X_test_s)
print(classification_report(y_test, y_pred, target_names=target_names))