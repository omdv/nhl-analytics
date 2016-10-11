#TODO: new season adds uncertainty tau
#TODO: vary beta

import pandas as pd
import numpy as np
import trueskill as ts
import xgboost as xgb
# from sqlalchemy import create_engine
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn import metrics as mt
from sklearn.cross_validation import train_test_split


np.random.seed(42)
# pd.options.display.float_format = '{:.4f}'.format

#preparing postgres engine
# engine = create_engine('postgresql://postgres:@192.168.99.100:5432/nhlstats')
# conn = engine.connect()

def read_dataset():
    print("Load csv...")
    df = pd.read_csv('team_stats_by_game.csv',parse_dates=['gameDate'])

    print('Process dataset...')
    df['gameType'],fct = pd.factorize(df.gameType)
    df = df.set_index('gameId').sort_values(by='gameDate')
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
def trueskill_forward_pass(df,season,n_iter=3):
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
        for g in np.sort(df.index.unique()):
            # Get team names and preGame stats
            game = df.ix[g]
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
            
            # Calculation
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
                team['gameId'] = g
                team['preMean'] = float(p['preRating'].mu)
                team['preStd'] = float(p['preRating'].sigma)
                team['postMean'] = float(p['postRating'].mu)
                team['postStd'] = float(p['postRating'].sigma)
                team['gameLoc'] = p['GameLoc']
                a = teams[p['teamAbbrev']]
                a.append(team.copy())
                teams[p['teamAbbrev']] = a

    # Convert to teams df
    k = list(teams)
    teams_df = pd.DataFrame(teams[k[0]])
    for t in k[1:]:
        teams_df = pd.concat([teams_df,pd.DataFrame(teams[t])])

    # Merge teams
    teams_home = teams_df[teams_df.gameLoc == 'H']
    del teams_home['gameLoc']
    del teams_home['gameDate']
    del teams_home['teamAbbrev']
    columns = teams_home.columns
    columns = [row+'Home' if row not in ['gameId','gameDate']\
               else row for row in columns]
    teams_home.columns = columns

    teams_road = teams_df[teams_df.gameLoc == 'R']
    del teams_road['gameLoc']
    del teams_road['gameDate']
    del teams_road['teamAbbrev']
    columns = teams_road.columns
    columns = [row+'Road' if row not in ['gameId'] else row for row in columns]
    teams_road.columns = columns

    teams = pd.merge(teams_home,teams_road,on='gameId')
    teams = pd.merge(df.reset_index(),teams,on='gameId').set_index('gameId')

    return teams_df, teams

## auxiliary function to get elo processed for several seasons
#def get_trueskill_seasons(seasons,params):
#    df = process_trueskill_forward(seasons[0],params)
#    for season in seasons[1:]:
#        df = pd.concat([df,process_trueskill_forward(season,params)])
#    return df

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

    print('XGBoost params: {}'.format(params))

    if valsize > 0:
        watchlist = [(dtrain, 'train'), (dvalid,'valid')]
    else:
        watchlist = [(dtrain, 'train')]

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,\
        early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    imp = get_importance(gbm, features)
    print('Importance array:\n{}'.format(imp))

    pred = gbm.predict(dtest, ntree_limit=gbm.best_iteration+1)
    print('\nPredicting test...')
    print('Matthews: {:.4f}'.format(mt.matthews_corrcoef(y_test,np.where(pred>0.5,1,0))))
    print('Accuracy: {:.4f}'.format(mt.accuracy_score(y_test,np.where(pred>0.5,1,0))))
    print('ROC: {:.4f}'.format(mt.roc_auc_score(y_test,pred)))

    return gbm, imp, pred

def get_importance(gbm, features):
    """
    Getting relative feature importance
    """
    importance = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
    importance = importance/1.e-2/importance.values.sum()
    return importance


if __name__ == '__main__':
    # TrueSkill params
    TSK_MEAN = 25.0
    TSK_SIGMA = TSK_MEAN/3.
    TSK_BETA = TSK_MEAN/6.
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

    df = read_dataset()

    season = 2011
    teams, dfs = trueskill_forward_pass(df,season,n_iter=1)

    dfs['winProb'] = dfs.apply(win_probability,axis=1)

    features=['winProb']
    gbm,imp,pred = run_single_xgboost(dfs,features,"winsHome",10,100)
