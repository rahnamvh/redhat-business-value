import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import random
from operator import itemgetter
import time
import copy
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

random.seed(2016)

def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

def intersect(a, b):
    return list(set(a) & set(b))

def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    output.remove('activity_id')
    return sorted(output)

def read_test_train():

    print("Read people.csv...")
    people = pd.read_csv("./input/people.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])

    print("Load train.csv...")
    train = pd.read_csv("./input/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    print("Load test.csv...")
    test = pd.read_csv("./input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

    print("Process tables...")
    for table in [train, test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table.drop('date', axis=1, inplace=True)
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            table['char_' + str(i)].fillna('type -999', inplace=True)
            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people.drop('date', axis=1, inplace=True)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    print("Merge...")
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(-999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(-999, inplace=True)

    features = get_features(train, test)
    return train, test, features

train, test, features = read_test_train()
test_size = 0.2
X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
print('Length train:', len(X_train.index))
print('Length valid:', len(X_valid.index))
y_train = X_train[target]
y_valid = X_valid[target]
dtrain = xgb.DMatrix(X_train[features], y_train)


def xgb_tuning(dtrain, X_valid, test, features):
    target='outcome'
    random_state=0
    eta = 0.1
    max_depth = 15
    subsample = 1.0
    colsample_bytree = 0.8
    min_child_weight = 5
    start_time = time.time()
    
    print('Starting time: {} '.format(time.asctime(time.localtime(time.time()))))
    
    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}, MIN_CHILD_WEIGHT: {}'.\
       format(eta, max_depth, subsample, colsample_bytree, min_child_weight))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "silent": 1,
        "seed": random_state,
             }
    num_boost_round = 1288
    gbm = xgb.train(params, dtrain, num_boost_round, verbose_eval=True)
    
    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]))
    
    score = roc_auc_score(X_valid[target].values, check)
    print('Check error value: {:.6f}'.format(score))
    
    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]))
    
    print('Importance array: ')
    feat_imp = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    
    return test_prediction.tolist(), score

def create_submission(score, test, prediction):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open("./submissions/" + sub_file, 'w')
    f.write('activity_id,outcome\n')
    total = 0
    for id in test['activity_id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

test_prediction, score = xgb_tuning(dtrain, X_valid, test, features)
create_submission(score, test, test_prediction)
