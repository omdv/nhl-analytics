import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.optimize import minimize

pd.options.mode.chained_assignment = None  #suppress chained assignment warning
pd.options.display.float_format = '{:.2f}'.format

#preparing postgres engine
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhl')
conn = engine.connect()

# process the database and calculate ELO rating for one season
def process_elo_forward(season,params):
    
    #ELO params
    ELO_DELTA = 200.
    ELO_MEAN = 1500.
    ELO_K_FACTOR = 16
    ELO_HOMEBONUS = params[0]
    ELO_OTWIN_COEFF = params[1]

    ts['preGameELO'] = -1.0
    ts['postGameELO'] = -1.0
    ts['gamePredict'] = -1.0
    ts['gameResult'] = -1.0
    ts['accuracy'] = 0.0

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
                dfs.preGameELO.ix[fg.name] = float(ELO_MEAN)
            initAssign = False
        
        # #get team names and pre-ELO
        hTeam = dfs.ix[g,'H'].teamAbbrev
        rTeam = dfs.ix[g,'R'].teamAbbrev
        hELO = dfs.ix[g,'H'].preGameELO
        rELO = dfs.ix[g,'R'].preGameELO

        # #expected outcome of the game
        dpower = (rELO - hELO - ELO_HOMEBONUS)/ELO_DELTA/2.
        expHome = 1.0/(1.0+np.power(10,dpower))
        dfs.gamePredict.ix[g,'H'] = 1.0/(1.0+np.power(10,dpower))
        dfs.gamePredict.ix[g,'R'] = 1.0/(1.0+np.power(10,-dpower))
        
        # #get goals for the home team
        homeWin = dfs.ix[g,'H'].wins
        roadWin = dfs.ix[g,'R'].wins
        homeOTLoss = dfs.ix[g,'H'].otLosses
        roadOTLoss = dfs.ix[g,'R'].otLosses

        # real score - home team
        if (homeWin == 1):
            realHome = 1.0
        elif (roadWin == 1):
            realHome = 0.0
        
        # check for OT:
        if (homeOTLoss == 1):
            realHome = 1.0-ELO_OTWIN_COEFF
        elif (roadOTLoss == 1):
            realHome = ELO_OTWIN_COEFF
        
        # real outcome of the game
        dfs.gameResult.ix[g,'H'] = realHome
        dfs.gameResult.ix[g,'R'] = 1.0 - realHome

        # post-game ELO rating
        deltaHome = ELO_K_FACTOR*(realHome-expHome)
        dfs.postGameELO.ix[g,'H'] = hELO + deltaHome
        dfs.postGameELO.ix[g,'R'] = rELO - deltaHome

        # calculate accuracy
        if (dfs.gameResult.ix[g,'H'] > 0.5) & (dfs.gamePredict.ix[g,'H'] > 0.5):
            dfs.accuracy.ix[g,'H'] = 1.0
            dfs.accuracy.ix[g,'R'] = 1.0
        elif (dfs.gameResult.ix[g,'H'] < 0.5) & (dfs.gamePredict.ix[g,'H'] < 0.5):
            dfs.accuracy.ix[g,'H'] = 1.0
            dfs.accuracy.ix[g,'R'] = 1.0

        
        # assign forward to a new preGameELO
        homeTeamFuture = dfs[(dfs.teamAbbrev == hTeam) &\
            (dfs.index.get_level_values('gameId') > g)]
        roadTeamFuture = dfs[(dfs.teamAbbrev == rTeam) &\
            (dfs.index.get_level_values('gameId') > g)]
        
        if len(homeTeamFuture) > 0:
            nhix = homeTeamFuture.iloc[0].name
            dfs.preGameELO.ix[nhix] = hELO + deltaHome
        
        if len(roadTeamFuture) > 0:
            nrix = roadTeamFuture.iloc[0].name
            dfs.preGameELO.ix[nrix] = rELO - deltaHome

    return dfs
    
# auxiliary function to get elo processed for several seasons
def get_elo_seasons(seasons,params):
    df = process_elo_forward(seasons[0],params)
    for season in seasons[1:]:
        df = pd.concat([df,process_elo_forward(season,params)])
    return df

# objective function for minimization
def elo_minimize_func(params):
    df = get_elo_seasons([2015],params)
    return (1.0-df.accuracy.mean())


if __name__ == '__main__':
    # read existing dataframe
    ts = pd.read_sql('team_stats_by_game',engine)
    
    params = np.array([63,0.69])
    params = np.array([50,0.76])
    # df = get_elo_seasons(np.arange(2005,2016),params)

    # minimize ELO parameters  
    res = minimize(elo_minimize_func, params, method='Nelder-Mead',tol=1.0e-3,
                    options={'disp': True, 'maxiter': 20})