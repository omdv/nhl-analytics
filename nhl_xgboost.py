from __future__ import print_function
import pandas as pd
import numpy as np
import time
import copy
import datetime
import numba
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score

#preparing postgres engine
engine = create_engine('postgresql://postgres:@192.168.99.100:5432/nhlstats')
conn = engine.connect()

np.random.seed(42)

def read_dataset():
    print("Load sql...")
    df = pd.read_sql('team_stats_public',engine,parse_dates=['gameDate'])

    print('Process dataset...')
    df['gameTypeHome'],fct = pd.factorize(df.gameTypeHome)

    return df

# --------------------------------------------------------#
# Processing functions

def slice_team(df,teamSlct,varCode,hist_len=10):
    newdf = {}
    cols = [teamSlct+'_'+varCode+'_L'+x for x in np.arange(1,hist_len+1)[::-1].astype('str')]
    for team in df[teamSlct].unique():
        hist_vals = df[df[teamSlct] == team][varCode].values
        hist_idx = df[df[teamSlct] == team].index
        for date in hist_idx:
            pos = np.searchsorted(hist_idx,date)
            slicer = hist_vals[:pos][-hist_len:]
            if len(slicer)<hist_len:
                slicer = np.insert(slicer,0,-999*np.ones(hist_len-len(slicer)))
            newdf[date] = pd.Series(slicer,index=cols)
    return(df.join(pd.DataFrame(newdf).T))

# --------------------------------------------------------#

def intersect(a, b):
    """
    For get_features() use
    """
    return list(set(a) & set(b))

# add derived features here
def derive_features(df):
    """
    This function allows making new features
    """
    print("Derive new features...")

    combos = [
        ['teamIdHome','goalsForHome'],
        ['teamIdHome','goalsAgainstHome'],
        ['teamIdHome','ppGoalsForHome'],
        ['teamIdHome','ppGoalsAgainstHome'],
        ['teamIdHome','shotsForHome'],
        ['teamIdHome','shotsAgainstHome'],
        ['teamIdHome','faceoffWinPctgHome'],
        ['teamIdHome','penaltyKillPctgHome'],
        ['teamIdHome','ppOpportunitiesHome'],
        ['teamIdHome','ppPctgHome'],
        ['teamIdHome','shNumTimesHome'],
        ['teamIdRoad','goalsForHome'],
        ['teamIdRoad','goalsAgainstHome'],
        ['teamIdRoad','ppGoalsForHome'],
        ['teamIdRoad','ppGoalsAgainstHome'],
        ['teamIdRoad','shotsForHome'],
        ['teamIdRoad','shotsAgainstHome'],
        ['teamIdRoad','faceoffWinPctgHome'],
        ['teamIdRoad','penaltyKillPctgHome'],
        ['teamIdRoad','ppOpportunitiesHome'],
        ['teamIdRoad','ppPctgHome'],
        ['teamIdRoad','shNumTimesHome']
    ]

    for combo in combos:
        df = slice_team(df,combo[0],combo[1])
        print(combo)
    return df

def get_features(df,testsize=20):
    """
    Get intersection of train and test as a list of features
    """
    mask = np.random.rand(df.shape[0]) < testsize/1.e2
    test = df[mask]
    train = df[~mask]

    features = intersect(train.columns, test.columns)
    correlated = ['winsHome',
        'faceoffWinPctgHome',
        'faceoffsLostHome',
        'faceoffsWonHome',
        'gameTypeHome',
        'goalsAgainstHome',
        'goalsForHome',
        'otPlayed',
        'penaltyKillPctgHome',
        'penaltyKillPctgRoad',
        'ppGoalsAgainstHome',
        'ppGoalsForHome',
        'ppOpportunitiesHome',
        'ppPctgHome',
        'ppPctgRoad',
        'seasonId',
        'shNumTimesHome',
        'shotsAgainstHome',
        'shotsForHome',
        'gameId',
        'gameDate',
        'teamAbbrevHome',
        'teamFullNameHome',
        'teamIdHome',
        'teamAbbrevRoad',
        'teamFullNameRoad',
        'teamIdRoad']
    features = [f for f in features if f not in correlated]
    return train,test,sorted(features)

# --------------------------------------------------------#

def run_single(train,test,features,target,valsize,num_boost_round):
    """
    Single xgboost run, returns predictions and validation score
    """

    # create a small validation set - unique people id
    if valsize > 0:
        mask = np.random.rand(train.shape[0]) < valsize/1.e2
        valid = train[mask]
        train = train[~mask]

        y_valid = valid[target]
        valid = valid[features]
    
    y_train = train[target]
    y_test = test[target]
    train = train[features]
    test = test[features]

    dtrain = xgb.DMatrix(train, label = y_train, missing = np.nan)
    dtest = xgb.DMatrix(test, missing = np.nan)

    if valsize > 0:
        dvalid = xgb.DMatrix(valid, label = y_valid, missing = np.nan)

    print('Starting xgboost cycle...')
    print('Shape of train: {}'.format(train.shape))
    print('Shape of test: {}'.format(test.shape))
    if valsize > 0:
        print('Shape of validation: {}'.format(valid.shape))

    # tree booster params
    early_stopping_rounds = 10
    start_time = time.time()
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eta": 0.1,
        "gamma": 0,
        "tree_method": 'exact',
        "max_depth": 8,
        "min_child_weight": 2,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "silent": 1,
        "seed": 42
    }

    print('XGBoost params: {}'.format(params))

    if valsize > 0:
        watchlist = [(dtrain, 'train'), (dvalid,'valid')]
    else:
        watchlist = [(dtrain, 'train')]

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,\
        early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    if valsize > 0:
        check = gbm.predict(dvalid, ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.values, check)
    else:
        check = gbm.predict(dtrain, ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_train.values, check)

    print('Crossval error: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array:\n{}'.format(imp))

    print("Predict test...")
    test_prediction = gbm.predict(dtest, ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_test.values, test_prediction)
    print('Test roc-auc: {:.6f}'.format(score))
    print('Test pred accuracy: {:.6f}'.
            format(np.sum(np.logical_and(y_test.values, test_prediction)))

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction, score, gbm, imp

def get_importance(gbm, features):
    """
    Getting relative feature importance
    """
    importance = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
    importance = importance/1.e-2/importance.values.sum()
    return importance

if __name__ == '__main__':
    df = read_dataset()
    df = derive_features(df)
    train,test,features = get_features(df)

    print('Shape of train: {}'.format(train.shape))
    print('Shape of test: {}'.format(test.shape))
    print('Features [{}]: {}'.format(len(features), sorted(features)[0:10]))

    prediction, score, model, importance =\
        run_single(train,test,features,'winsHome',15,300)
