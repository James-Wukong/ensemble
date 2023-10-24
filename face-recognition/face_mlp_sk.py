from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Preprocesing
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
# Models
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
import skorch
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

mlp_param_grid = {'hidden_layer_sizes':
        # Configurations with 1 hidden layer:
        [(100, ), (50, ), (25, ), (10, ), (5, ),
        # Configurations with 2 hidden layers:
        (100, 50, ), (50, 25, ), (25, 10, ), (10, 5, ),
        # Configurations with 3 hidden layers:
        (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )],
    'activation': ['relu'], 
    'solver': ['adam'], 
    'alpha': [0.0001]
}
mlp_cv = make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=1500),
            param_grid=mlp_param_grid,
            cv=5, 
            n_jobs=-1
        )
    )
t0 = time()
mlp_cv.fit(X_train, y_train)
print(f'done in {time() - t0:.3f}s')
print('Best params found by grid search:')
print(mlp_cv.steps[-1][1].best_params_)
y_pred = mlp_cv.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))