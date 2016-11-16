import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import trueskill as ts
import matplotlib.pyplot as plt
import xgboost as xgb
import datetime as dt
import seaborn as sns
import pickle
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn import metrics as mt
from sklearn.cross_validation import train_test_split
from sqlalchemy import create_engine
from scraper_schedule import get_schedule_by_dates

np.random.seed(42)
pd.options.display.float_format = '{:.2f}'.format

#preparing postgres engine
engine = create_engine('postgresql://postgres:@192.168.99.100:5432/nhlstats')

def read_dataset():
    print("Load dataset...")
    df = pd.read_sql('team_stats_by_game',engine,parse_dates=['gameDate'])

    # choose only nhl games
    df = df[(df.gameType=='R') | (df.gameType=='P')]

    df['dtindex'] = pd.DatetimeIndex(df.gameDate)
    startDate = df[df.gameId==2005020001].dtindex
    df['timeShift'] = df.apply(lambda row: (row.dtindex-startDate), axis=1)
    df['timeShift'] = df.apply(lambda row: row['timeShift'].days, axis=1)

    print('Process dataset...')
    df['gameType'],fct = pd.factorize(df.gameType)
    df = df.sort_values(by='gameId')
    return df

def merge_with_schedule(df,sch):
    print("Merging with new games...")

    df = pd.concat([df,sch])

    del df['dtindex']
    df['dtindex'] = pd.DatetimeIndex(df.gameDate)
    startDate = df.gameDate.min()
    df['timeShift'] = df.apply(lambda row: (row.dtindex-startDate), axis=1)
    df['timeShift'] = df.apply(lambda row: row['timeShift'].days, axis=1)    
    df = df.reset_index().sort_values(by='gameDate')
    return df

def v_function(t,e):
    denom = norm.cdf(t-e)
    if denom < 2.222758749e-162:
        if t < 0.0:
            return 1.0
        else:
            return 0.0
    return norm.pdf(t-e)/denom

def w_function(t,e):
    v = v_function(t,e)
    return v*(v+t-e)

def win_probability(row):
    delta_mu = row.preMeanHome - row.preMeanRoad + TSK_HOME_BONUS
    denom = np.sqrt(2* (env.beta**2) + row.preStdHome**2 + row.preStdRoad**2)
    return norm.cdf(delta_mu / denom)

# process to generate separate df with trueskill
def trueskill_forward_pass(df,season,n_iter=1):
    df = df[df.seasonId == season]
    # initialize containers for the results
    teams = {}
    lastTeamRating = {}

    for it in np.arange(n_iter):

        # initialize
        if it == 0:
            for t in df.teamAbbrevHome.unique():
                lastTeamRating[t] = {'mu':TSK_MEAN,'sigma':TSK_SIGMA}
                teams[t] = []
        else:
            for t in df.teamAbbrevHome.unique():
                lastTeamRating[t] = {
                    'mu':teams[t][-1]['postMean'],
                    'sigma':teams[t][-1]['postStd']}
                teams[t] = []

        # Forward pass
        for gId in np.sort(df.index.unique()):
            # Get team names and preGame stats
            game = df.ix[gId]
            hTeam = game.teamAbbrevHome
            rTeam = game.teamAbbrevRoad
            hMean = lastTeamRating[hTeam]['mu']
            hSigma = lastTeamRating[hTeam]['sigma']
            rMean = lastTeamRating[rTeam]['mu']
            rSigma = lastTeamRating[rTeam]['sigma']

            # Get game outcome
            winner = {}
            loser = {}
            if (game.winsHome):
                winner['teamAbbrev'] = hTeam
                winner['preRating'] = ts.Rating(mu=hMean,sigma=hSigma)
                winner['GameLoc'] = 'H'
                loser['teamAbbrev'] = rTeam
                loser['preRating'] = ts.Rating(mu=rMean,sigma=rSigma)
                loser['GameLoc'] = 'R'
            else:
                winner['teamAbbrev'] = rTeam
                winner['preRating'] = ts.Rating(mu=rMean,sigma=rSigma)
                winner['GameLoc'] = 'R'
                loser['teamAbbrev'] = hTeam
                loser['preRating'] = ts.Rating(mu=hMean,sigma=hSigma)
                loser['GameLoc'] = 'H'
            
            # Update TrueSkill
            winner['postRating'],loser['postRating'] =\
                ts.rate_1vs1(winner['preRating'],loser['preRating'],env=env)

            # Update result
            for p in [winner,loser]:
                # update lastTeamRating
                lastTeamRating[p['teamAbbrev']] = {
                    'mu':float(p['postRating'].mu),
                    'sigma':float(p['postRating'].sigma) }
                team = {}
                team['teamAbbrev'] = p['teamAbbrev']
                team['gameDate'] = game.gameDate
                team['gameId'] = gId
                team['preMean'] = float(p['preRating'].mu)
                team['preStd'] = float(p['preRating'].sigma)
                team['postMean'] = float(p['postRating'].mu)
                team['postStd'] = float(p['postRating'].sigma)
                team['gameLoc'] = p['GameLoc']
                a = teams[p['teamAbbrev']]
                a.append(team.copy())
                teams[p['teamAbbrev']] = a

    # Convert result to dataframe
    k = list(teams)
    teams_as_df = pd.DataFrame(teams[k[0]])
    for t in k[1:]:
        teams_as_df = pd.concat([teams_as_df,pd.DataFrame(teams[t])])

    return teams_as_df.sort_values(by='gameId')

def merge_teams_with_df(df,teams):
    """
    Expecting full df and teams df
    Merged df as a result
    """

    # Merge teams
    teams_home = teams[teams.gameLoc == 'H']
    del teams_home['gameLoc']
    del teams_home['gameDate']
    del teams_home['teamAbbrev']
    columns = [row+'Home' if row not in ['gameId','gameDate']\
               else row for row in teams_home.columns]
    teams_home.columns = columns

    teams_road = teams[teams.gameLoc == 'R']
    del teams_road['gameLoc']
    del teams_road['gameDate']
    del teams_road['teamAbbrev']
    columns = [row+'Road' if row not in ['gameId']\
                else row for row in teams_road.columns]
    teams_road.columns = columns

    teams = pd.merge(teams_home,teams_road,on='gameId')
    df = pd.merge(df.reset_index(),teams,on='gameId').set_index('gameId')
    return df

def proces_trueskill_forward():
    # TrueSkill params
    TSK_MEAN = 1200.0
    TSK_SIGMA = TSK_MEAN/3.
    TSK_BETA = TSK_MEAN/2.
    TSK_TAU = TSK_MEAN/300.
    TSK_DRAW_PROB = 0.0
    TSK_HOME_BONUS = TSK_MEAN/10.

    # Trueskill setup
    env=ts.TrueSkill(
        mu=TSK_MEAN,
        sigma=TSK_SIGMA,
        beta=TSK_BETA,
        tau=TSK_TAU,
        draw_probability=TSK_DRAW_PROB)

    season = 2011

    # tms = trueskill_forward_pass(df,season,n_iter=1)
    # dfs = merge_teams_with_df(df,tms)
    # dfs = get_rest_days(dfs)

    # dfs['winProb'] = dfs.apply(win_probability,axis=1)
    # print('ROC: {:.4f}'.format(mt.roc_auc_score(dfs.winsHome,dfs.winProb)))
    return 0

def get_rest_days(df):
    """
    Expecting full as input df
    """
    for team in df.teamAbbrevHome.unique():
        timeDelta = df[(df.teamAbbrevHome == team) |\
            (df.teamAbbrevRoad == team)].gameDate.diff()
        # add off-season
        timeDelta.iloc[0] = dt.timedelta(days=120)
        # timeDelta = [x.days for x in timeDelta]
        df.loc[(df.teamAbbrevHome == team),'restDaysHome']=timeDelta
        df.loc[(df.teamAbbrevRoad == team),'restDaysRoad']=timeDelta
    
    df['restDaysHome'] = df.apply(lambda x: x.restDaysHome.days,axis=1)
    df['restDaysRoad'] = df.apply(lambda x: x.restDaysRoad.days,axis=1)

    return df

def run_single_xgboost(train,features,target,valsize,num_boost_round):
    """
    Single xgboost run, returns predictions and validation score
    """
    train, test = train_test_split(train, test_size=0.33, random_state=42)

    # create a small validation set - unique people id
    if valsize > 0:
        mask = np.random.rand(train.shape[0]) < valsize/1.e2
        valid = train[mask]
        train = train[~mask]

        y_valid = valid[target]
        valid = valid[features]

    y_train = train[target]
    train = train[features]
    y_test = test[target]
    test = test[features]

    dtrain = xgb.DMatrix(train, label = y_train, missing = np.nan)
    dtest = xgb.DMatrix(test, label = y_test, missing = np.nan)

    if valsize > 0:
        dvalid = xgb.DMatrix(valid, label = y_valid, missing = np.nan)

    print('Starting xgboost cycle...')
    if valsize > 0:
        print('Shape of validation: {}'.format(valid.shape))

    # tree booster params
    early_stopping_rounds = 10
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eta": 0.1,
        "early_stopping_rounds":10,
        "gamma": 0,
        "tree_method": 'exact',
        "max_depth": 8,
        "min_child_weight": 2,
        "subsample": 0.7,
        "colsample_bytree": 1.0,
        "silent": 1,
        "seed": 42,
        "eval_metric": "auc"
    }

    if valsize > 0:
        watchlist = [(dtrain, 'train'), (dvalid,'valid')]
    else:
        watchlist = [(dtrain, 'train')]

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,\
        early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

    imp = get_importance(gbm, features)
    print('Importance array:\n{}'.format(imp))

    pred = gbm.predict(dtest, ntree_limit=gbm.best_iteration+1)
    return gbm, imp, y_test, pred

def get_evaluation(true,pred,ths):
    print('\nEvaluating test...')
    pred = np.where(pred>ths,1,0)
    print('AUC: {:.4f}'.format(mt.roc_auc_score(true,pred)))
    print('Matthews: {:.4f}'.format(mt.matthews_corrcoef(true,pred)))
    print('Accuracy: {:.4f}'.format(mt.accuracy_score(true,pred)))
    print('Precision: {:.4f}'.format(mt.precision_score(true,pred)))
    print('Recall: {:.4f}'.format(mt.recall_score(true,pred)))
    print(mt.confusion_matrix(true,pred,labels=[0,1]))


def get_importance(gbm, features):
    """
    Getting relative feature importance
    """
    importance = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
    importance = importance/1.e-2/importance.values.sum()
    return importance

def save_for_infernet(df):
    # df = df.reset_index().set_index('gameDate')

    fields = [
        'teamIdHome',
        'teamIdRoad',
        'seasonId',
        'winsHome',
        'timeShift',
        'gameId'
    ]
    short = df[fields].sort_values(by='gameId')
    short['timeShift'],fct = pd.factorize(short.timeShift)

    with open('timeShift.fct','wb') as fp:
        pickle.dump(fct,fp)

    short.to_csv('trueskill_in.csv',index=False)
    print("Saved for Infer.net: {0}".format(short.timeShift.max()+1))
    return short.timeShift.max()+1

def merge_with_infernet(df):
    tsk = pd.read_csv('trueskill_out.csv',sep="|",header=None)
    means = tsk.applymap(lambda x: float(str(x)[9:-1].split(",")[0]))
    means.index = means.index
    means = means.stack().reset_index()
    means.columns = ['timeShift','teamId','tsMean']    

    for key in ['teamIdHome','teamIdRoad']:
        df = pd.merge(df,means,left_on=['timeShift',key],right_on=['timeShift','teamId'],how='left')
        del df['teamId']
        df.rename(columns={'tsMean':key+'Mean'}, inplace=True)

    std = tsk.applymap(lambda x: float(str(x)[9:-1].split(",")[1]))
    std = std.stack().reset_index()
    std.columns = ['timeShift','teamId','tsMean']
    std.tsMean = np.sqrt(std.tsMean)

    for key in ['teamIdHome','teamIdRoad']:
        df = pd.merge(df,std,left_on=['timeShift',key],right_on=['timeShift','teamId'],how='left')
        del df['teamId']
        df.rename(columns={'tsMean':key+'Std'}, inplace=True)
        df[key+'Std'] = np.sqrt(df[key+'Std'])

    return df,means,std

# function to get trueskill by team
def get_tsk_by_team(df,window=10):
    print("Calculating trueskill evolution for every team")
    meanByTeam = {}
    stdByTeam = {}
    for team in df.teamIdHome.unique():
        meanByTeam[team] = pd.concat([df[(df.teamIdHome==team)].teamIdHomeMean,
            df[df.teamIdRoad==team].teamIdRoadMean]).sort_index()
        stdByTeam[team] = pd.concat([df[(df.teamIdHome==team)].teamIdHomeStd,
            df[df.teamIdRoad==team].teamIdRoadStd]).sort_index()
    meanByTeam = pd.DataFrame(meanByTeam)
    stdByTeam = pd.DataFrame(stdByTeam)
    # meanByTeam = meanByTeam.rolling(window=window,min_periods=1).mean()
    # stdByTeam = stdByTeam.rolling(window=window,min_periods=1).mean()
    return meanByTeam,stdByTeam

# calculate win probability
def win_prob_2(row,homeBonusMean,homeBonusStd):
    delta_mu = row.homeMean - row.roadMean + homeBonusMean
    denom = np.sqrt(row.homeStd**2 + row.roadStd**2 + homeBonusStd)
    return norm.cdf(delta_mu / denom)

# get optimal cutoff using roc_curve
def get_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = mt.roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

if __name__ == '__main__':

    df = read_dataset()
    
    # MANUAL PORTION
    # --------
    save_for_infernet(df)
    # df,means,stds = merge_with_infernet(df)
    # --------

    # # merge with schedule
    # today = dt.date.today().strftime('%Y-%m-%d')
    # sch = get_schedule_by_dates(today,today)
    # df = merge_with_schedule(df,sch)

    # # pivot means and std by team
    # print("Pivot infer.net output by team...")
    # meansByTeam = means.pivot(index='timeShift',columns='teamId',values='tsMean')
    # stdsByTeam = stds.pivot(index='timeShift',columns='teamId',values='tsMean')

    # print("Shift skill forward...")
    # missingDays = df[df.winsHome.isnull()].timeShift.unique()
    # meansByTeam.ix(meansByTeam.index.max()+1,:) = np.nan
    # stdsByTeam.ix(stdsByTeam.index.max()+1,:) = np.nan


    # df['winProb'] = df.apply(lambda row: win_prob_2(row,131, 127.4),axis=1)
    # df = get_rest_days(df)

    # features=[
    #     'homeMean',
    #     'homeStd',
    #     'roadMean',
    #     'roadStd',
    #     'restDaysHome',
    #     'restDaysRoad'
    # ]
    
    # gbm,imp,true,pred = run_single_xgboost(df[df.seasonId>2013],features,"winsHome",10,100)
    # threshold = get_optimal_cutoff(true,pred)
    # get_evaluation(true,pred,threshold)




