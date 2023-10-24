from time import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Preprocesing
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
# Models
import sklearn.svm as svm
# import utils
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

svm_cv = make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(svm.SVC(class_weight='balanced'),
            {'kernel': ['poly', 'rbf'], 
             'C': 10. ** np.arange(-2, 3)},
            # {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 4)},
            cv=5, 
            n_jobs=-1
        )
    )
t0 = time()
svm_cv.fit(X_train, y_train)
print(f'done in {time() - t0:.3f}s')
print('Best params found by grid search:')

print(svm_cv.steps[-1][1].best_params_)
y_pred = svm_cv.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))