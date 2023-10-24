from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
# Transforming string Target to an int
encoder = LabelEncoder()
binary_encoded_y = pd.Series(encoder.fit_transform(y))

#Train Test Split
train_X, test_X, train_y, test_y = train_test_split(X, binary_encoded_y, random_state=2020)
boosting_clf_ada_boost= AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=3
    )
bagging_clf_rf = RandomForestClassifier(n_estimators=200, max_depth=1,random_state=2020)
clf_rf = RandomForestClassifier(n_estimators=200, max_depth=1,random_state=2020)
clf_ada_boost = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1,random_state=2020),
        n_estimators=3
    )
clf_logistic_reg = LogisticRegression(solver='liblinear',random_state=2020)

#Customizing and Exception message
class NumberOfClassifierException(Exception):
    pass

#Creating a stacking class
class Stacking():
    '''
    We suppose that at least the First N-1 Classifiers have
    a predict_proba function.
    '''
    def __init__(self, classifiers):
        if(len(classifiers) < 2):
            raise NumberOfClassifierException("You must fit your classifier with 2 classifiers at least");
        else:
            self._classifiers = classifiers

    def fit(self,data_X,data_y):
        stacked_data_X = data_X.copy()
        for classfier in self._classifiers[:-1]:
            classfier.fit(data_X,data_y)
            stacked_data_X = np.column_stack((stacked_data_X,classfier.predict_proba(data_X)))
        last_classifier = self._classifiers[-1]
        last_classifier.fit(stacked_data_X,data_y)

    def predict(self,data_X):
        stacked_data_X = data_X.copy()
        for classfier in self._classifiers[:-1]:
            prob_predictions = classfier.predict_proba(data_X)
            stacked_data_X = np.column_stack((stacked_data_X,prob_predictions))
        last_classifier = self._classifiers[-1]
        return last_classifier.predict(stacked_data_X)
    
bagging_clf_rf.fit(train_X, train_y)
boosting_clf_ada_boost.fit(train_X, train_y)
classifers_list = [clf_rf,clf_ada_boost,clf_logistic_reg]

clf_stacking = Stacking(classifers_list)
clf_stacking.fit(train_X,train_y)

predictions_bagging = bagging_clf_rf.predict(test_X)
predictions_boosting = boosting_clf_ada_boost.predict(test_X)
predictions_stacking = clf_stacking.predict(test_X)

print("For Bagging : F1 Score {}, Accuracy {}"
      .format(round(f1_score(test_y,predictions_bagging),2),
              round(accuracy_score(test_y,predictions_bagging),2)
            )
        )
print("For Boosting : F1 Score {}, Accuracy {}"
      .format(round(f1_score(test_y,predictions_boosting),2),
              round(accuracy_score(test_y,predictions_boosting),2)
            )
        )
print("For Stacking : F1 Score {}, Accuracy {}"
      .format(round(f1_score(test_y,predictions_stacking),2),
              round(accuracy_score(test_y,predictions_stacking),2)
            )
        )