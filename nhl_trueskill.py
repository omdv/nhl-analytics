import pandas as pd
import numpy as np
import trueskill as ts
# from sqlalchemy import create_engine
from scipy.optimize import minimize
from scipy.stats import norm

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

# def rating_1v1(wMean,wStd,lMean,lStd):
    # # Get game outcome
    # if (game.winsHome):
    #     wTeam = hTeam
    #     lTeam = rTeam
    #     homeBonus = TSK_HOME_BONUS
    #     wPreMean = hMean
    #     wPreStd = hStd
    #     lPreMean = rMean
    #     lPreStd = rStd
    # else:
    #     wTeam = rTeam
    #     lTeam = hTeam
    #     homeBonus = -TSK_HOME_BONUS
    #     wPreMean = rMean
    #     wPreStd = rStd
    #     lPreMean = hMean
    #     lPreStd = hStd
    # TSK_DRAW_MARGIN = norm.ppf(0.5*(TSK_DRAW_PROB+1.))*np.sqrt(1.+1.)*TSK_BETA
    # totStd = np.sqrt(2*(TSK_BETA**2)+hStd**2+rStd**2)
    # meanDelta = wPreMean-lPreMean+homeBonus
    # TSK_v = v_function(meanDelta/totStd,TSK_DRAW_MARGIN/totStd)
    # TSK_w = w_function(meanDelta/totStd,TSK_DRAW_MARGIN/totStd)
    # wMean = wPreMean + (wPreStd**2+TSK_TAU**2)/totStd*TSK_v
    # lMean = lPreMean - (lPreStd**2+TSK_TAU**2)/totStd*TSK_v
    
    # wVar = wPreStd**2+TSK_TAU**2
    # wStd = np.sqrt(wVar*(1-TSK_w*wVar/totStd**2))
    # lVar = lPreStd**2+TSK_TAU**2
    # lStd = np.sqrt(lVar*(1-TSK_w*lVar/totStd**2))
    # team = {}
    # team['teamAbbrev'] = wTeam
    # team['gameDate'] = game.gameDate
    # team['gameId'] = str(g)
    # team['skillMean'] = float(wMean)
    # team['skillStd'] = float(wStd)
    # a = teams[wTeam]
    # a.append(team.copy())
    # teams[wTeam] = a
    
    # team = {}
    # team['teamAbbrev'] = lTeam
    # team['gameDate'] = game.gameDate
    # team['gameId'] = str(g)
    # team['skillMean'] = float(lMean)
    # team['skillStd'] = float(lStd)
    # a = teams[lTeam]
    # a.append(team.copy())
    # teams[lTeam] = a

def win_probability(home, road):
    delta_mu = home.mu - road.mu + TSK_HOME_BONUS
    denom = np.sqrt(2* (env.beta**2) + home.sigma**2 + road.sigma**2)
    return norm.cdf(delta_mu / denom)

# # processing df inline
# def process_trueskill_forward(df,season):
#     df['preMeanHome'] = TSK_MEAN
#     df['preSigmaHome'] = TSK_SIGMA
#     df['preMeanRoad'] = TSK_MEAN
#     df['preSigmaRoad'] = TSK_SIGMA
#     df['postMeanHome'] = -1.0
#     df['postSigmaHome'] = -1.0
#     df['postMeanRoad'] = -1.0
#     df['postSigmaRoad'] = -1.0
#     df['homeWinProb'] = -1.0
#     dfs = df[df.seasonId == season]
#     #loop over games
#     for g in np.sort(df.index.unique()):
        
#         # Get team names and preGame stats
#         game = df.ix[g]
#         hTeam = game['teamAbbrevHome']
#         rTeam = game['teamAbbrevRoad']
#         hMean = game['preMeanHome']
#         hStd = game['preSigmaHome']
#         rMean = game['preMeanRoad']
#         rStd = game['preSigmaRoad']
#         '''
#         ------------------------------
#         True Skill Calculation section
#         ------------------------------
#         '''
#         # Get game outcome
#         if (game.winsHome):
#             wCode = 'Home'
#             lCode = 'Road'
#             wTeam = hTeam
#             lTeam = rTeam
#             winner = ts.Rating(mu=hMean,sigma=hStd)
#             loser = ts.Rating(mu=rMean,sigma=rStd)
#         else:
#             wCode = 'Road'
#             lCode = 'Home'
#             wTeam = rTeam
#             lTeam = hTeam
#             loser = ts.Rating(mu=hMean,sigma=hStd)
#             winner = ts.Rating(mu=rMean,sigma=rStd)
#         winner,loser = ts.rate_1vs1(winner,loser,env=env)
#         # Assign the result
#         dfs.ix[g,'postMean'+wCode] = winner.mu
#         dfs.ix[g,'postSigma'+wCode] = winner.sigma
#         dfs.ix[g,'postMean'+lCode] = loser.mu
#         dfs.ix[g,'postSigma'+lCode] = loser.sigma
#         # # Assign forward the preGame
#         # winnerFuture = dfs[(dfs.teamAbbrev == dfs.ix[g,winner].teamAbbrev) &\
#         #     (dfs.index.get_level_values('gameId') > g)]
#         # loserFuture = dfs[(dfs.teamAbbrev == dfs.ix[g,loser].teamAbbrev) &\
#         #     (dfs.index.get_level_values('gameId') > g)]
        
#         # if len(winnerFuture) > 0:
#         #     nhix = winnerFuture.iloc[0].name
#         #     dfs.preGameTSKmean.ix[nhix] = wMean
#         #     dfs.preGameTSKsigma.ix[nhix] = wSigma
        
#         # if len(loserFuture) > 0:
#         #     nrix = loserFuture.iloc[0].name
#         #     dfs.preGameTSKmean.ix[nrix] = lMean
#         #     dfs.preGameTSKsigma.ix[nrix] = lSigma
#         # Expected outcome of the game
#         # if hMEAN > rMEAN:
#         #     dfs.gamePredictTSK.ix[g,'H'] = 1.0
#         #     dfs.gamePredictTSK.ix[g,'R'] = 0.0
#         # elif hMEAN < rMEAN:
#         #     dfs.gamePredictTSK.ix[g,'H'] = 0.0
#         #     dfs.gamePredictTSK.ix[g,'R'] = 1.0
#         # else:
#         #     dfs.gamePredictTSK.ix[g,'H'] = 0.5
#         #     dfs.gamePredictTSK.ix[g,'R'] = 0.5
#         # nSigma = np.sqrt(hSigma**2+rSigma**2)
#         # nMean = meanDelta
#         # dfs.gameResultProbTSK.ix[g,winner] = 1 - norm.cdf(0,loc=nMean,scale=nSigma)
#         # dfs.gameResultProbTSK.ix[g,loser] = norm.cdf(0,loc=nMean,scale=nSigma)
#         # Calculate accuracy
#         # if (dfs.gameResultTSK.ix[g,'H'] > 0.5) &\
#         #     (dfs.gamePredictTSK.ix[g,'H'] > 0.5):
#         #     dfs.accuracyTSK.ix[g,'H'] = 1.0
#         #     dfs.accuracyTSK.ix[g,'R'] = 1.0
#         # elif (dfs.gameResultTSK.ix[g,'H'] < 0.5) &\
#         #     (dfs.gamePredictTSK.ix[g,'H'] < 0.5):
#         #     dfs.accuracyTSK.ix[g,'H'] = 1.0
#         #     dfs.accuracyTSK.ix[g,'R'] = 1.0
#     return dfs

# process to generate separate df with trueskill
def process_trueskill_forward(df,season):
    df = df[df.seasonId == season]
    
    # initialize containers for the results
    teams = {}
    lastTeamRating = {}
    teamList = df.teamAbbrevHome.unique()

    for t in teamList:
        lastTeamRating[t] = {'mu':TSK_MEAN,'sigma':TSK_SIGMA}
        teams[t] = []

    # for t in teamList:
    #     team = {}
    #     team['teamAbbrev'] = t
    #     team['preMean'] = float(TSK_MEAN)
    #     team['preStd'] = float(TSK_SIGMA)
    #     firstHomeGame = df[df.teamAbbrevHome == t].iloc[0]
    #     firstRoadGame = df[df.teamAbbrevRoad == t].iloc[0]
    #     if firstHomeGame.gameDate < firstRoadGame.gameDate:
    #         team['gameDate'] = firstHomeGame.gameDate
    #         team['gameLoc'] = 'H'
    #         team['gameId'] = firstHomeGame.name
    #     else:
    #         team['gameDate'] = firstRoadGame.gameDate
    #         team['gameLoc'] = 'R'
    #         team['gameId'] = firstRoadGame.name
    #     a = []
    #     a.append(team.copy())
    #     teams[t] = a

    # Forward pass
    for g in np.sort(df.index.unique()):
        # Get team names and preGame stats
        game = df.ix[g]
        hTeam = game.teamAbbrevHome
        rTeam = game.teamAbbrevRoad
        # hMean = teams[hTeam][-1]['preMean']
        # hStd = teams[hTeam][-1]['preStd']
        # rMean = teams[rTeam][-1]['preMean']
        # rStd = teams[rTeam][-1]['preStd']
        hMean = lastTeamRating[hTeam]['mu']
        hSigma = lastTeamRating[hTeam]['sigma']
        rMean = lastTeamRating[rTeam]['mu']
        rSigma = lastTeamRating[rTeam]['sigma']

        '''
        True Skill Calculation section
        '''
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
        print(winner,loser)

        # update result
        for p in [winner,loser]:
            team = {}
            team['teamAbbrev'] = p['teamAbbrev']
            team['gameDate'] = game.gameDate
            team['gameId'] = str(g)
            team['preMean'] = float(p['preRating'].mu)
            team['preStd'] = float(p['preRating'].sigma)
            team['postMean'] = float(p['postRating'].mu)
            team['postStd'] = float(p['postRating'].sigma)
            team['gameLoc'] = p['GameLoc']
            # a = teams[p['teamAbbrev']]
            # a.append(team.copy())
            teams[p['teamAbbrev']] = team.copy()
           
    # convert to df
    k = list(teams)
    tf = pd.DataFrame(teams[k[0]])
    for t in k[1:]:
        tf = pd.concat([tf,pd.DataFrame(teams[t])])
    return tf

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
    tsk = process_trueskill_forward(df,season)
