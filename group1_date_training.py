import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import datetime
import time

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

def find_prediction(train, test, features, target):
    X_train, X_valid = train_test_split(train, test_size=0.2, random_state=0)
    
    simp_df = X_train.groupby(features)[target].mean()
    simp_df_new = simp_df.add_suffix('').reset_index()
    
    for ft in features:
        simp_df_new[ft] = simp_df_new[ft].astype("int")
        
    train_predict = pd.merge(X_train, simp_df_new, on = features, how = 'left')
    valid_predict = pd.merge(X_valid, simp_df_new, on = features, how = 'left')
    
    auc_train = roc_auc_score(train_predict[target + '_x'].values, train_predict[target + '_y'].values)
    auc_valid = roc_auc_score(valid_predict[target + '_x'].values, valid_predict[target + '_y'].values)
    
    test_predict = pd.merge(test, simp_df_new, on = features, how = 'left')
    
    return test_predict, auc_valid

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

train, test, features = read_test_train()
test_prediction1, auc_score1 = find_prediction(train, test, ['group_1', 'year_x', 'month_x', 'day_x'], 'outcome')
test_prediction2, auc_score2 = find_prediction(train, test, ['group_1'], 'outcome')
xgboostout = pd.read_csv("./submissions/submission_0.998135479953_2016-09-09-02-48.csv")
tp1 = test_prediction1[['activity_id', 'outcome']]
tp2 = test_prediction2[['activity_id', 'outcome']]
tp3 = pd.merge(tp2, xgboostout, how='inner', on='activity_id')
tp3['outcome'] = (tp3['outcome_x'] + tp3['outcome_y'])/2 
tp3.drop('outcome_x', axis=1, inplace=True)
tp3.drop('outcome_y', axis=1, inplace=True)
tp4 = pd.merge(tp1, tp3, how='inner', on='activity_id')
tp4['outcome'] = tp4.outcome_x.combine_first(tp4.outcome_y)
tp4.drop('outcome_x', axis=1, inplace=True)
tp4.drop('outcome_y', axis=1, inplace=True)
tp5 = pd.merge(tp4, xgboostout, how='inner', on='activity_id')
tp5['outcome'] = tp5.outcome_x.combine_first(tp5.outcome_y)
tp5.drop('outcome_x', axis=1, inplace=True)
tp5.drop('outcome_y', axis=1, inplace=True)
test_prediction = tp5['outcome'].tolist()
create_submission((auc_score1 + auc_score2)/2, test, test_prediction)
