import pandas as pd
import numpy as np
import trueskill as ts
# from sqlalchemy import create_engine
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# pd.options.display.float_format = '{:.4f}'.format
# pd.options.display.int_format = '{:.4f}'.format

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
def process_trueskill_forward(df,season):
    df = df[df.seasonId == season]
    
    # initialize containers for the results
    teams = {}
    lastTeamRating = {}

    # initiate
    for t in df.teamAbbrevHome.unique():
        lastTeamRating[t] = {'mu':TSK_MEAN,'sigma':TSK_SIGMA}
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
        
        # calculation
        winner['postRating'],loser['postRating'] =\
            ts.rate_1vs1(winner['preRating'],loser['preRating'],env=env)

        # update result
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

    # convert to teams df
    k = list(teams)
    teams_df = pd.DataFrame(teams[k[0]])
    for t in k[1:]:
        teams_df = pd.concat([teams_df,pd.DataFrame(teams[t])])

    # merge teams
    teams_home = teams_df[teams_df.gameLoc == 'H']
    del teams_home['gameLoc']
    del teams_home['gameDate']
    del teams_home['teamAbbrev']
    columns = teams_home.columns
    columns = [row+'Home' if row not in ['gameId','gameDate'] else row for row in columns]
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

# auxiliary function to get elo processed for several seasons
def get_trueskill_seasons(seasons,params):
    df = process_trueskill_forward(seasons[0],params)
    for season in seasons[1:]:
        df = pd.concat([df,process_trueskill_forward(season,params)])
    return df

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
    teams, dfs = process_trueskill_forward(df,season)

    dfs['winProb'] = dfs.apply(win_probability,axis=1)

