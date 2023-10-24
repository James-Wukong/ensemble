from time import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Preprocesing
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
# Models
import sklearn.linear_model as lm
# import utils
from src.utils import image_utils as iu
from src.utils import init

device = init.init_device()
# download the data
lfw_people = init.init_people()
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = init.init_sample(lfw_people)

# split data into training and test set
X_train, X_test, y_train, y_test = init.init_data(lfw_people, n_samples)

target_names = lfw_people.target_names
n_classes = target_names.shape[0]

lrl2_cv = make_pipeline(
preprocessing.StandardScaler(),
# preprocessing.MinMaxScaler(), # Would have done the job either
GridSearchCV(lm.LogisticRegression(max_iter=1000, class_weight='balanced',
                                   fit_intercept=False),
                                    {'C': 10. ** np.arange(-3, 3)},
                                    cv=5, n_jobs=5)
                                )
t0 = time()
lrl2_cv.fit(X=X_train, y=y_train)
print(f'done in {time() - t0:.3f}s')
print('Best params found by grid search:')
print(lrl2_cv.steps[-1][1].best_params_)
y_pred = lrl2_cv.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

coefs = lrl2_cv.steps[-1][1].best_estimator_.coef_
coefs = coefs.reshape(-1, h, w)
iu.plot_gallery(coefs, target_names, h, w)