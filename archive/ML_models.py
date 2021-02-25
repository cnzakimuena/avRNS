# Check the versions of libraries

# Python version
import sys
# print('Python: {}'.format(sys.version))
# scipy
import scipy
# print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
# print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
# print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
# print('sklearn: {}'.format(sklearn.__version__))
import conf_test

# Load libraries
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from os.path import join as p_join

# Load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pandas.read_csv(url, names=names)

# Load dataset
str_dataPath = r'C:\Users\cnzak\Desktop\data\avRNS\biophotonics'
dirPath_str = p_join(str_dataPath, 'df_input.csv').replace("\\", "/")  # combine directory strings
dataset = pd.read_csv(dirPath_str)
num_rec = 600  # define # of records per subject

# shape
print(dataset.shape)

# head
print(dataset.head(dataset.shape[0]))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# # box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# plt.show()

# # histograms
# dataset.hist()
# plt.show()

# # scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

# Split-out validation dataset
# # split by records
# array = dataset.values
# dataCol_dim = dataset.shape[1]-1
# X = array[:, 0:dataCol_dim]
# Y = array[:, dataCol_dim]
# validation_size = 0.20
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
#                                                                                 random_state=seed)
# identity confounding split
X_train, X_validation, Y_train, Y_validation = conf_test.permutations(dataset, num_rec)

# Test options and evaluation metric
seed = None
# scoring = 'accuracy'
scoring = 'roc_auc'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=3, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# # Make predictions on validation dataset
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#
# # Make predictions on validation dataset
# rf_model = DecisionTreeClassifier()
# rf_model.fit(X_train, Y_train)
# predictions = rf_model.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
