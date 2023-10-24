import pandas as pd
import numpy as np
from matplotlib import pyplot

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# get the classification dataset
def get_dataset(data_type='classification'):
    if data_type == 'classification':
        X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    elif data_type == 'regression':
         X, y = make_regression(n_samples=10000, n_features=20, n_informative=15, noise=0.1, random_state=1)
    return X, y

def const_models(model_type='classification') -> dict:
	models = {
		'classification': {
			'lr': LogisticRegression(),
            'knn':  KNeighborsClassifier(),
            'cart': DecisionTreeClassifier(),
            'svm': SVC(),
            'bayes': GaussianNB(),
        },
		'regression': {
			'knn':  KNeighborsRegressor(),
            'cart': DecisionTreeRegressor(),
            'svm': SVR(),
        },
    }
	
	return models[model_type]

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    for key, model in const_models().items():
        level0.append((key, model))
		
    # define meta learner model, 
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# get a list of classification models to evaluate
def get_models(model_type='classification'):
    models = const_models(model_type)
    models['stacking'] = get_stacking()
    return models


# evaluate a given model using cross-validation
def evaluate_cls_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# evaluate a given model using cross-validation
def evaluate_reg_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# define dataset
X, y = get_dataset()

# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_cls_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()


# define dataset
X, y = get_dataset('regression')

# get the models to evaluate
models = get_models('regression')
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_reg_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()
