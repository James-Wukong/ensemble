import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",names=names)
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

num_trees = 100
max_features = 3

kfold = model_selection.KFold(n_splits=10)

# Bagged Decision Trees for Classification
rf = DecisionTreeClassifier(max_features=max_features)
model = BaggingClassifier(estimator=rf, n_estimators=num_trees, random_state=2020)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()))

# Bagged Decision Trees for Random Forest Classification
# rf = DecisionTreeClassifier()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, random_state=2020)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()))