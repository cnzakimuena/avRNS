# Check the versions of libraries

# Python version
import sys
# print('Python: {}'.format(sys.version))
# scipy
import scipy
# print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
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
import matplotlib.pyplot as plt


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

# PART 1 perform 20 iterations (permutations) of test set, each with 3 k-fold

tot_results_0 = []
tot_results_1 = []
num_iterations = 20
for q in range(num_iterations):

    # each time append 3 results array to cumulative array (3 * 20 = 60)

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

    results_array_0 = results[0]
    tot_results_0 = np.concatenate([tot_results_0, results_array_0])
    results_array_1 = results[1]
    tot_results_1 = np.concatenate([tot_results_1, results_array_1])

# PART 2 perform 1 iteration (split only) of test set, each with 3 k-fold
    # each time append array to cumulative array

# Split-out validation dataset
# split by records
array = dataset.values
dataCol_dim = dataset.shape[1]-1
X = array[:, 0:dataCol_dim]
Y = array[:, dataCol_dim]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
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

mean_results_array_0 = results[0]
mean_results_0 = mean_results_array_0.mean()
mean_results_array_1 = results[1]
mean_results_1 = mean_results_array_1.mean()

# PART 3 show confluence test results
# Generate two sets of numbers from a normal distribution
# one with mean = 4 sd = 0.5, another with mean (loc) = 1 and sd (scale) = 2
randomSet = np.random.normal(loc=4, scale=0.5, size=1000)
anotherRandom = np.random.normal(loc=1, scale=2, size=1000)
# Define a Figure and Axes object using plt.subplots
# Axes object is where we do the actual plotting (i.e. draw the histogram)
# Figure object is used to configure the actual figure (e.g. the dimensions of the figure)
fig, axs = plt.subplots(2)
# Plot a histogram with custom-defined bins, with a blue colour, transparency of 0.4
# Plot the density rather than the raw count using normed = True
axs[0].hist(tot_results_0, bins=np.arange(0, 1, 0.02), color='#134a8e', alpha=0.4)
# Plot solid line for the means
axs[0].axvline(mean_results_0, color='blue')
# # Plot dotted lines for the std devs
# axs[0].axvline(np.mean(randomSet) - np.std(randomSet), linestyle='--', color='blue')
# axs[0].axvline(np.mean(randomSet) + np.std(randomSet), linestyle='--', color='blue')
# Set the title, x- and y-axis labels
axs[0].set_xlabel("AUC, LR")
axs[0].set_ylabel("Density")

axs[1].hist(tot_results_1, bins=np.arange(0, 1, 0.02), color='#e8291c', alpha=0.4)
axs[1].axvline(mean_results_0, color='red')
# axs[1].axvline(np.mean(anotherRandom) - np.std(anotherRandom), linestyle='--', color='red')
# axs[1].axvline(np.mean(anotherRandom) + np.std(anotherRandom), linestyle='--', color='red')
axs[1].set_xlabel("AUC, KNN")
axs[1].set_ylabel("Density")

plt.suptitle("OCT spatial series spectrum, RPEb-BM and CC, \n 22 subjects")
# Set the Figure's size as a 5in x 5in figure
fig.set_size_inches((5, 5))

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

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
