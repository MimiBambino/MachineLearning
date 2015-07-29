#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA


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
                 'total_stock_value']

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

for person in my_dataset:
    my_dataset[person]['to_poi_ratio_normalized'] = 'NaN'
    my_dataset[person]['from_poi_ratio_normalized'] = 'NaN'
    if my_dataset[person]['to_poi_ratio'] != "NaN":
        my_dataset[person]['to_poi_ratio_normalized'] =  (my_dataset[person]['to_poi_ratio'] - avg_to_poi_ratio)
    if my_dataset[person]['from_poi_ratio'] != "NaN":
        my_dataset[person]['from_poi_ratio_normalized'] =  (my_dataset[person]['from_poi_ratio'] - avg_to_poi_ratio)

features_list.append('to_poi_ratio_normalized')
features_list.append('from_poi_ratio_normalized')

# Remove less important features
sparse_data = {}
remove_dict = {}
for name in data_dict:
    for feat in data_dict[name]:
        if data_dict[name][feat] == "NaN":
            if feat in sparse_data:
                sparse_data[feat] += 1
                # If more than 108 are NaN, I want to remove these features
                if sparse_data[feat] > (144*.7):
                    remove_dict[feat] = sparse_data[feat]
            else:
                sparse_data[feat] = 1

print "Features with 70% 'NaN' values : "
print remove_dict
print "\n"

# for key, value in data_dict.iteritems():
#   if data_dict[key]['loan_advances'] != "NaN":
#        print key, data_dict[key]['loan_advances']

count = 0
for key, value in data_dict.iteritems():
    if data_dict[key]['poi'] == True and data_dict[key]['director_fees'] != "NaN":
        print key, data_dict[key]['director_fees']
    else:
        count += 1

if count == 145:
    print "No POIs have Director's Fees"

count = 0
for key, value in data_dict.iteritems():
    if data_dict[key]['poi'] == True and data_dict[key]['restricted_stock_deferred'] != "NaN":
        print key, data_dict[key]['restricted_stock_deferred']
    else:
        count += 1

if count == 145:
    print "No POIs have Restricted Stock Deferred"

features_list.remove('director_fees')
print "Director Fees removed from features_list.", "\n"
features_list.remove('restricted_stock_deferred')
print "Restricted Stock Deferred removed from features_list.", "\n"
features_list.remove('loan_advances')
print "Loan Advances removed from features_list.", "\n"
features_list.remove('deferral_payments')
print "Deferral Payments removed from features_list.", "\n"

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Select the 11 best features with SelectKBest
from sklearn.feature_selection import SelectKBest
print "SelectKBest Feature Ranking"
k_best = SelectKBest(k=10)
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

# Scaler, if needed.
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

features = scaler.fit_transform(features)


### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.feature_selection import SelectKBest

########## Number 1 ##########
from sklearn.neighbors import KNeighborsClassifier
#Accuracy: 0.87060 Precision: 0.52502  Recall: 0.30950 F1: 0.38943 F2: 0.33718
estimators = [('k-best', SelectKBest(k=9)),
              ('k-neighbors', KNeighborsClassifier(n_neighbors = 3,
                                                   algorithm = 'ball_tree',
                                                   weights ='distance',
                                                   p = 5))]

########## Number 2 ##########
#from sklearn.naive_bayes import GaussianNB
#Accuracy: 0.85013  Precision: 0.42584  Recall: 0.35600 F1: 0.38780F2: 0.36807
#estimators = [('k-best', SelectKBest(k = 5)), ('naive_bayes', GaussianNB())]

########## Number 3 ##########
# Accuracy: 0.81800 Precision: 0.31823  Recall: 0.31950 F1: 0.31886F2: 0.31924
#clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'random')

###### Insufficient Classifiers #####
# from sklearn.grid_search import GridSearchCV
# parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#               'weights': ('uniform', 'distance'),
#               'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
#               'p':[1,2] }
# knc = KNeighborsClassifier()
# clf = GridSearchCV(knc, parameters)

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2), n_estimators=30, learning_rate = .8)

#clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', splitter='random'), learning_rate = .8)

# from sklearn.linear_model import LogisticRegression
# estimators = [('k-best', SelectKBest(k=5)), ('log', LogisticRegression())]

# Accuracy: 0.86380 Precision: 0.48065  Recall: 0.26700 F1: 0.34330 F2: 0.29305
#clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree', weights='distance', p=5)

clf = Pipeline(estimators)
###########################

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
