from time import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Preprocesing
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
# Models
import sklearn.linear_model as lm
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

anova_l2lr = Pipeline([('standardscaler', preprocessing.StandardScaler()),
            ('anova', SelectKBest(f_classif)),
            ('l2lr', lm.LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced',
            fit_intercept=False))
        ]
    )
param_grid = {'anova__k': [50, 100, 500, 1000, 1500, X_train.shape[1]],
        'l2lr__C': 10. ** np.arange(-3, 3)}
anova_l2lr_cv = GridSearchCV(anova_l2lr, cv=5, param_grid=param_grid, n_jobs=-1)
t0 = time()
anova_l2lr_cv.fit(X=X_train, y=y_train)
print(f'done in {time() - t0:.3f}s')
print("Best params found by grid search:")
print(anova_l2lr_cv.best_params_)
y_pred = anova_l2lr_cv.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))