import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.optimize import minimize
from scipy.stats import norm

# pd.options.display.float_format = '{:.4f}'.format
# pd.options.display.int_format = '{:.4f}'.format


#preparing postgres engine
engine = create_engine('postgresql://postgres:@192.168.99.100:5432/nhlstats')
conn = engine.connect()

def read_dataset():
    print("Load sql...")
    df = pd.read_sql('team_stats_public',engine,parse_dates=['gameDate'])

    print('Process dataset...')
    df['gameTypeHome'],fct = pd.factorize(df.gameTypeHome)

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

# process the database and calculate ELO rating for one season
def process_trueskill_forward(df,season,params):

    # TrueSkill params
    TSK_MEAN = 25.0
    TSK_SIGMA = TSK_MEAN/3.
    TSK_BETA = params[0] #TSK_MEAN/6
    TSK_DYNAMICS_FACTOR = params[1] #TSK_MEAN/300
    TSK_DRAW_PROB = 0.0
    TSK_HOME_BONUS = params[2] #TSK_MEAN/10
    TSK_DRAW_MARGIN = norm.ppf(0.5*(TSK_DRAW_PROB+1))*\
                        np.sqrt(1 + 1)*TSK_BETA

    df = df[df.seasonId == season]

    # initialize containers for the results
    teams = {}
    teamList = df.teamAbbrevHome.unique()
    for t in teamList:
        team = {}
        team['teamAbbrev'] = t
        team['skillMean'] = float(TSK_MEAN)
        team['skillStd'] = float(TSK_SIGMA)

        firstHomeGame = df[df.teamAbbrevHome == t].iloc[0]
        firstRoadGame = df[df.teamAbbrevRoad == t].iloc[0]

        if firstHomeGame.gameDate < firstRoadGame.gameDate:
            team['gameDate'] = firstHomeGame.gameDate
            # team['gameId'] = 
        else:
            team['gameDate'] = firstRoadGame.gameDate
            # team['gameId'] = 

        a = []
        a.append(team.copy())
        teams[t] = a

    # Forward pass
    for g in np.sort(df.index.unique()):

        # Get team names and preGame stats
        game = df.ix[g]
        hTeam = game.teamAbbrevHome
        rTeam = game.teamAbbrevRoad
        hMean = teams[hTeam][-1]['skillMean']
        hStd = teams[hTeam][-1]['skillStd']
        rMean = teams[rTeam][-1]['skillMean']
        rStd = teams[rTeam][-1]['skillStd']

        # Get game outcome
        if (game.winsHome):
            wTeam = hTeam
            lTeam = rTeam
            homeBonus = TSK_HOME_BONUS
            wPreMean = hMean
            wPreStd = hStd
            lPreMean = rMean
            lPreStd = rStd
        else:
            wTeam = rTeam
            lTeam = hTeam
            homeBonus = -TSK_HOME_BONUS
            wPreMean = rMean
            wPreStd = rStd
            lPreMean = hMean
            lPreStd = hStd

        # Update skill
        totStd = np.sqrt(2*(TSK_BETA**2)+hStd**2+rStd**2)
        meanDelta = wPreMean-lPreMean+homeBonus

        # v and w functions
        TSK_v = v_function(meanDelta/totStd,TSK_DRAW_MARGIN/totStd)
        TSK_w = w_function(meanDelta/totStd,TSK_DRAW_MARGIN/totStd)

        # Mean change
        wMean = wPreMean + (wPreStd**2+TSK_DYNAMICS_FACTOR**2)/totStd*TSK_v
        lMean = lPreMean - (lPreStd**2+TSK_DYNAMICS_FACTOR**2)/totStd*TSK_v
        
        # Sigma change
        wVar = wPreStd**2+TSK_DYNAMICS_FACTOR**2
        wStd = np.sqrt(wVar*(1-TSK_w*wVar/totStd**2))

        lVar = lPreStd**2+TSK_DYNAMICS_FACTOR**2
        lStd = np.sqrt(lVar*(1-TSK_w*lVar/totStd**2))

        # Update - add the new game record or change the one if the first game
        team = {}
        team['teamAbbrev'] = wTeam
        team['gameDate'] = game.gameDate
        team['gameId'] = str(g)
        team['skillMean'] = float(wMean)
        team['skillStd'] = float(wStd)
        a = teams[wTeam]
        a.append(team.copy())
        teams[wTeam] = a
        
        team = {}
        team['teamAbbrev'] = lTeam
        team['gameDate'] = game.gameDate
        team['gameId'] = str(g)
        team['skillMean'] = float(lMean)
        team['skillStd'] = float(lStd)
        a = teams[lTeam]
        a.append(team.copy())
        teams[lTeam] = a

        # nSigma = np.sqrt(hSigma**2+rSigma**2)
        # nMean = meanDelta
        # dfs.gameResultProbTSK.ix[g,winner] = 1 - norm.cdf(0,loc=nMean,scale=nSigma)
        # dfs.gameResultProbTSK.ix[g,loser] = norm.cdf(0,loc=nMean,scale=nSigma)

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
    df = read_dataset()

    season = 2011
    params = [25./6.,0./300.,0./10.]
    ts = process_trueskill_forward(df,season,params)



    
    # season = 2011
    # params = [4.15,8.428e-1,2.53] #[4.15,8.428e-2,2.53] optimized - 2006,2010,2013
    # df = get_trueskill_seasons(np.arange(2005,2007),params)

