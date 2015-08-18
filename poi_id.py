#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
#from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
#from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.metrics import precision_score, recall_score, accuracy_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'bonus',
                 'deferral_payments',
                 'deferred_income',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'loan_advances',
                 'long_term_incentive',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'salary',
                 'total_payments',
                 'total_stock_value',
                 'other']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers

# Remove observations that are not people.
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
# Remove person with 'NaN' values for every feature.
data_dict.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

# New features to epresent the proportion of emails to or from a POI.
for person in my_dataset:
    if my_dataset[person]['to_messages'] == 'NaN' or my_dataset[person]['from_this_person_to_poi'] == 'NaN':
        my_dataset[person]['to_poi_ratio'] = 'NaN'
    else:
        my_dataset[person]['to_poi_ratio'] = float(my_dataset[person]['from_this_person_to_poi'])/float(my_dataset[person]['to_messages'])

for person in my_dataset:
    my_dataset[person]['from_poi_ratio'] = 'NaN'
    if my_dataset[person]['from_messages'] != 'NaN' and my_dataset[person]['from_poi_to_this_person'] != 'NaN':
        my_dataset[person]['from_poi_ratio'] = float(my_dataset[person]['from_poi_to_this_person'])/float(my_dataset[person]['from_messages'])

features_list.append('to_poi_ratio')
print "To POI Ratio added to features_list.", "\n"
features_list.append('from_poi_ratio')
print "From POI Ratio added to features_list.", "\n"

num_to_poi_ratios = 0
total_to_poi_ratios = 0
num_from_poi_ratios = 0
total_from_poi_ratios = 0
for person in my_dataset:
    if my_dataset[person]['to_poi_ratio'] != "NaN":
        num_to_poi_ratios += 1
        total_to_poi_ratios += my_dataset[person]['to_poi_ratio']
    if my_dataset[person]['from_poi_ratio'] != "NaN":
        num_from_poi_ratios += 1
        total_from_poi_ratios += my_dataset[person]['from_poi_ratio']

avg_to_poi_ratio = total_to_poi_ratios / float(num_to_poi_ratios)

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scaler, if needed.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Select the 11 best features with SelectKBest
from sklearn.feature_selection import SelectKBest
print "SelectKBest Feature Ranking"
k_best = SelectKBest(k=11)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(), features_list[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
count = 1
refined_features = []
for i in results_list:
    if i[0]:
        print count, "\t", i
        refined_features.append(i[1])
        count += 1

# Ranked 11 best features as determined by SelectKFeatures
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'bonus',
                 'salary',
                 'deferred_income', #5
                 'long_term_incentive', # 6
                 'restricted_stock',# 7
                 'total_payments', # 8
                 'loan_advances',  # 9
                 'expenses',       # 10
                 'from_poi_ratio'  # 11
                 ]

########## FINAL FEATURES LIST #########
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'bonus']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scaler, for K Nearest Neighbors.
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

## For local testing
def print_results(i):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42)
    clf.fit(features_train, labels_train)
    print 'Best score: %0.3f' % clf.best_score_
    print 'Best parameters set:'
    best_parameters = clf.best_estimator_.get_params()
    new_params = {}
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
        new_params[param_name] = best_parameters[param_name]
    predictions = clf.predict(features_test)
    # print 'Accuracy: ', accuracy_score(labels_test, predictions)
    # print 'Precision: ', precision_score(labels_test, predictions)
    # print 'Recall: ', recall_score(labels_test, predictions)
    print "----------------------------------------------------------------------"

clf_list = ["tree", "bayes", "adaboost", "kNeighbors"]

for i in clf_list:
    print "---------------------------"+ i.upper() +"----------------------------"
    if i == "tree":
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier()
        parameters = {'criterion': ["gini", "entropy"],
              'splitter': ['best', 'random'],
              'min_samples_split': [2,3,4,5]
             }
        clf = GridSearchCV(tree, parameters, verbose=1, cv=10)
        print_results(i)
    if i == "bayes":
        from sklearn.naive_bayes import GaussianNB
        bayes =  GaussianNB()
        parameters = {}
        clf = GridSearchCV(bayes, parameters, verbose=1, cv=10)
        print_results(i)
    if i == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        adaboost = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', min_samples_split=3, splitter='best'))
        parameters = {'n_estimators': [10, 20, 30, 40, 50, 60, 70],
              'algorithm': ['SAMME', 'SAMME.R'],
              'learning_rate': [.5,.8, 1, 1.2, 1.5]}
        clf = GridSearchCV(adaboost, parameters, verbose=1, cv=10)
        print_results(i)
    if i == "kNeighbors":
        from sklearn.neighbors import KNeighborsClassifier
        kNeighbors = KNeighborsClassifier()
        parameters = {'n_neighbors': [2,3,4,5,6,7],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                      'weights': ['uniform', 'distance'],
                      'p': [3,4,5,6,7,8]
                     }
        clf = GridSearchCV(kNeighbors, parameters, verbose=1, cv=10)
        print_results(i)



########### BACKUP CLASSIFIERS ###########
#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, splitter='best')

#from sklearn.naive_bayes import GaussianNB
#clf =  GaussianNB()

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=2, p=6, weights='uniform')

###########################

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


########### FINAL CLASSIFIER ##############
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', min_samples_split=3, splitter='best'),
    algorithm='SAMME', learning_rate=1.5, n_estimators=40)

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
