import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.optimize import minimize
from scipy.stats import norm
# from math import erf, sqrt, pi, exp

pd.options.mode.chained_assignment = None  #suppress chained assignment warning
pd.options.display.float_format = '{:.2f}'.format

#preparing postgres engine
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhl')
conn = engine.connect()

# # process the database and calculate ELO rating for one season
# def process_trueskill_forward(season,params):
    
 

#     return dfs
    
# # auxiliary function to get elo processed for several seasons
# def get_trueskill_seasons(seasons,params):
#     df = process_elo_forward(seasons[0],params)
#     for season in seasons[1:]:
#         df = pd.concat([df,process_elo_forward(season,params)])
#     return df

# # objective function for minimization
# def trueskill_minimize_func(params):
#     df = get_elo_seasons([2015],params)
#     return (1.0-df.accuracy.mean())

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


if __name__ == '__main__':
    # read existing dataframe
    ts = pd.read_sql('team_stats_by_game',engine)
    season = 2012

    # TrueSkill params
    TSK_MEAN = 25.0
    TSK_SIGMA = TSK_MEAN/3
    TSK_BETA = TSK_MEAN/6
    TSK_DYNAMICS_FACTOR = TSK_MEAN/300
    TSK_DRAW_PROB = 0.1

    TSK_DRAW_MARGIN = norm.ppf(0.5*(TSK_DRAW_PROB+1))*\
                        np.sqrt(1 + 1)*TSK_BETA

    ts['preGameTSKmean'] = -1.0
    ts['preGameTSKsigma'] = -1.0
    ts['postGameTSKmean'] = -1.0
    ts['postGameTSKsigma'] = -1.0
    ts['gamePredictTSK'] = -1.0
    ts['gameResultTSK'] = -1.0
    ts['accuracyTSK'] = 0.0

    dfs = ts[ts.seasonId == season].set_index(['gameId','gameLocationCode'])
    
    #loop over games - after init
    gameList = dfs.index.levels[0].unique()
    teamList = dfs.teamAbbrev.unique()
    initAssign = True
    for g in gameList:
        
        #assign initial value at the beginning of the season
        if initAssign: #do only once
            for team in teamList:
                teamGames = dfs[dfs.teamAbbrev == team]      
                fg = teamGames.iloc[0]
                dfs.preGameTSKmean.ix[fg.name] = float(TSK_MEAN)
                dfs.preGameTSKsigma.ix[fg.name] = float(TSK_SIGMA)
            initAssign = False
        
        # Get team names and preGame stats
        hMEAN = dfs.ix[g,'H'].preGameTSKmean
        hSIGMA = dfs.ix[g,'H'].preGameTSKsigma
        rMEAN = dfs.ix[g,'R'].preGameTSKmean
        rSIGMA = dfs.ix[g,'R'].preGameTSKsigma

        # Get game outcome
        # TODO: Introduce Overtime logic
        if (dfs.ix[g,'H'].gameResultELO > dfs.ix[g,'R'].gameResultELO):
            winner = 'H'
            loser = 'R'
        else:
            winner = 'R'
            loser = 'H'

        # Trueskill calculation
        ccoef = np.sqrt(2*pow(TSK_BETA,2)+pow(hSIGMA,2)+pow(rSIGMA,2))
        MeanDelta = dfs.ix[g,winner].preGameTSKmean-dfs.ix[g,loser].preGameTSKmean

        # v and w functions
        TSK_v = v_function(MeanDelta/ccoef,TSK_DRAW_MARGIN/ccoef)
        TSK_w = w_function(MeanDelta/ccoef,TSK_DRAW_MARGIN/ccoef)

        # Updating values
        # Mean
        wMean = dfs.ix[g,winner].preGameTSKmean +\
            (dfs.ix[g,winner].preGameTSKsigma**2+TSK_DYNAMICS_FACTOR**2)/ccoef*TSK_v
        lMean = dfs.ix[g,loser].preGameTSKmean -\
            (dfs.ix[g,loser].preGameTSKsigma**2+TSK_DYNAMICS_FACTOR**2)/ccoef*TSK_v
        
        # Sigma
        wSigma = dfs.ix[g,winner].preGameTSKsigma**2+TSK_DYNAMICS_FACTOR**2
        wSigma = np.sqrt(wSigma*(1-TSK_w*wSigma/ccoef**2))

        lSigma = dfs.ix[g,loser].preGameTSKsigma**2+TSK_DYNAMICS_FACTOR**2
        lSigma = np.sqrt(lSigma*(1-TSK_w*lSigma/ccoef**2))

        dfs.postGameTSKmean.ix[g,winner] = wMean
        dfs.postGameTSKmean.ix[g,loser] = lMean
        dfs.postGameTSKsigma.ix[g,winner] = wSigma
        dfs.postGameTSKsigma.ix[g,loser] = lSigma

        # Assign forward
        winnerFuture = dfs[(dfs.teamAbbrev == dfs.ix[g,winner].teamAbbrev) &\
            (dfs.index.get_level_values('gameId') > g)]
        loserFuture = dfs[(dfs.teamAbbrev == dfs.ix[g,loser].teamAbbrev) &\
            (dfs.index.get_level_values('gameId') > g)]
        
        if len(winnerFuture) > 0:
            nhix = winnerFuture.iloc[0].name
            dfs.preGameTSKmean.ix[nhix] = wMean
            dfs.preGameTSKsigma.ix[nhix] = wSigma
        
        if len(loserFuture) > 0:
            nrix = loserFuture.iloc[0].name
            dfs.preGameTSKmean.ix[nrix] = lMean
            dfs.preGameTSKsigma.ix[nrix] = lSigma


        # # #expected outcome of the game
        # dpower = (rELO - hELO - ELO_HOMEBONUS)/ELO_DELTA/2.
        # expHome = 1.0/(1.0+pow(10,dpower))
        # dfs.gamePredict.ix[g,'H'] = 1.0/(1.0+pow(10,dpower))
        # dfs.gamePredict.ix[g,'R'] = 1.0/(1.0+pow(10,-dpower))
        
        # # #get goals for the home team
        # homeWin = dfs.ix[g,'H'].wins
        # roadWin = dfs.ix[g,'R'].wins
        # homeOTLoss = dfs.ix[g,'H'].otLosses
        # roadOTLoss = dfs.ix[g,'R'].otLosses

        # # real score - home team
        # if (homeWin == 1):
        #     realHome = 1.0
        # elif (roadWin == 1):
        #     realHome = 0.0
        
        # # check for OT:
        # if (homeOTLoss == 1):
        #     realHome = 1.0-ELO_OTWIN_COEFF
        # elif (roadOTLoss == 1):
        #     realHome = ELO_OTWIN_COEFF
        
        # # real outcome of the game
        # dfs.gameResult.ix[g,'H'] = realHome
        # dfs.gameResult.ix[g,'R'] = 1.0 - realHome

        # # post-game ELO rating
        # deltaHome = ELO_K_FACTOR*(realHome-expHome)
        # dfs.postGameELO.ix[g,'H'] = hELO + deltaHome
        # dfs.postGameELO.ix[g,'R'] = rELO - deltaHome

        # # calculate accuracy
        # if (dfs.gameResult.ix[g,'H'] > 0.5) & (dfs.gamePredict.ix[g,'H'] > 0.5):
        #     dfs.accuracy.ix[g,'H'] = 1.0
        #     dfs.accuracy.ix[g,'R'] = 1.0
        # elif (dfs.gameResult.ix[g,'H'] < 0.5) & (dfs.gamePredict.ix[g,'H'] < 0.5):
        #     dfs.accuracy.ix[g,'H'] = 1.0
        #     dfs.accuracy.ix[g,'R'] = 1.0

        
        # # assign forward to a new preGameELO
        # homeTeamFuture = dfs[(dfs.teamAbbrev == hTeam) &\
        #     (dfs.index.get_level_values('gameId') > g)]
        # roadTeamFuture = dfs[(dfs.teamAbbrev == rTeam) &\
        #     (dfs.index.get_level_values('gameId') > g)]
        
        # if len(homeTeamFuture) > 0:
        #     nhix = homeTeamFuture.iloc[0].name
        #     dfs.preGameELO.ix[nhix] = hELO + deltaHome
        
        # if len(roadTeamFuture) > 0:
        #     nrix = roadTeamFuture.iloc[0].name
        #     dfs.preGameELO.ix[nrix] = rELO - deltaHome
    