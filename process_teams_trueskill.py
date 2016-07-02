import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.optimize import minimize
from scipy.stats import norm

pd.options.mode.chained_assignment = None  #suppress chained assignment warning
pd.options.display.float_format = '{:.3f}'.format

#preparing postgres engine
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhl')
conn = engine.connect()


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

# process the database and calculate ELO rating for one season
def process_trueskill_forward(season,params):

    # TrueSkill params
    TSK_MEAN = 25.0
    TSK_SIGMA = TSK_MEAN/3
    TSK_BETA = params[0] #TSK_MEAN/6
    TSK_DYNAMICS_FACTOR = params[1] #TSK_MEAN/300
    TSK_DRAW_PROB = 0.0
    TSK_HOME_BONUS = params[2] #TSK_MEAN/10
    TSK_DRAW_MARGIN = norm.ppf(0.5*(TSK_DRAW_PROB+1))*\
                        np.sqrt(1 + 1)*TSK_BETA

    ts['preGameTSKmean'] = -1.0
    ts['preGameTSKsigma'] = -1.0
    ts['postGameTSKmean'] = -1.0
    ts['postGameTSKsigma'] = -1.0
    ts['gameResultProbTSK'] = -1.0
    ts['gamePredictTSK'] = -1.0
    ts['gameResultTSK'] = ts['gameResultELO']
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
        hMean = dfs.ix[g,'H'].preGameTSKmean+TSK_HOME_BONUS
        hSigma = dfs.ix[g,'H'].preGameTSKsigma
        rMean = dfs.ix[g,'R'].preGameTSKmean
        rSigma = dfs.ix[g,'R'].preGameTSKsigma

        # Get game outcome
        # TODO: Introduce Overtime logic
        if (dfs.ix[g,'H'].gameResultELO > dfs.ix[g,'R'].gameResultELO):
            winner = 'H'
            loser = 'R'
            homeBonus = TSK_HOME_BONUS
        else:
            winner = 'R'
            loser = 'H'
            homeBonus = -TSK_HOME_BONUS

        # UPDATE SKILL
        ccoef = np.sqrt(2*(TSK_BETA**2)+hSigma**2+rSigma**2)
        meanDelta = dfs.ix[g,winner].preGameTSKmean-dfs.ix[g,loser].preGameTSKmean+homeBonus

        # v and w functions
        TSK_v = v_function(meanDelta/ccoef,TSK_DRAW_MARGIN/ccoef)
        TSK_w = w_function(meanDelta/ccoef,TSK_DRAW_MARGIN/ccoef)

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


        # Expected outcome of the game
        # if hMEAN > rMEAN:
        #     dfs.gamePredictTSK.ix[g,'H'] = 1.0
        #     dfs.gamePredictTSK.ix[g,'R'] = 0.0
        # elif hMEAN < rMEAN:
        #     dfs.gamePredictTSK.ix[g,'H'] = 0.0
        #     dfs.gamePredictTSK.ix[g,'R'] = 1.0
        # else:
        #     dfs.gamePredictTSK.ix[g,'H'] = 0.5
        #     dfs.gamePredictTSK.ix[g,'R'] = 0.5
        nSigma = np.sqrt(hSigma**2+rSigma**2)
        nMean = meanDelta
        dfs.gameResultProbTSK.ix[g,winner] = 1 - norm.cdf(0,loc=nMean,scale=nSigma)
        dfs.gameResultProbTSK.ix[g,loser] = norm.cdf(0,loc=nMean,scale=nSigma)

        # Calculate accuracy
        # if (dfs.gameResultTSK.ix[g,'H'] > 0.5) &\
        #     (dfs.gamePredictTSK.ix[g,'H'] > 0.5):
        #     dfs.accuracyTSK.ix[g,'H'] = 1.0
        #     dfs.accuracyTSK.ix[g,'R'] = 1.0
        # elif (dfs.gameResultTSK.ix[g,'H'] < 0.5) &\
        #     (dfs.gamePredictTSK.ix[g,'H'] < 0.5):
        #     dfs.accuracyTSK.ix[g,'H'] = 1.0
        #     dfs.accuracyTSK.ix[g,'R'] = 1.0
    return dfs

# auxiliary function to get elo processed for several seasons
def get_trueskill_seasons(seasons,params):
    df = process_trueskill_forward(seasons[0],params)
    for season in seasons[1:]:
        df = pd.concat([df,process_trueskill_forward(season,params)])
    return df

# objective function for minimization
def trueskill_minimize(params):
    df = get_trueskill_seasons([2006,2010,2013],params)
    return (1.0-df.accuracyTSK.mean())


if __name__ == '__main__':
    ts = pd.read_sql('team_stats_by_game',engine)
    
    # season = 2011
    # params = [25./6.,25./300.,0.0,25./10.]
    params = [4.15,8.428e-1,2.53] #[4.15,8.428e-2,2.53] optimized - 2006,2010,2013
    df = get_trueskill_seasons(np.arange(2005,2007),params)


    # minimize parameters  
    # params0 = np.array([25./6.,25./300.,0.0,25./10.])
    # res = minimize(trueskill_minimize, params0, method='Nelder-Mead',
                    # options={'disp': True, 'maxiter': 50})