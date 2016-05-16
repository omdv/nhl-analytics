import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

pd.options.mode.chained_assignment = None  #suppress chained assignment warning

#postgres
engine = create_engine(
    'postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()


def process_elo_forward(season):
    
    dfs = df[df.seasonId == season]      
    teamList = dfs.teamAbbrev.unique()
    
    #assign initial values
    for team in teamList:
        teamGames = dfs[(dfs.teamAbbrev == team) |\
            (dfs.opponentTeamAbbrev == team)].\
                sort_values(by='gameDate')
        #first game of the season
        fg = teamGames.iloc[0]
        if fg.teamAbbrev == team:
            dfs[dfs.gameId == fg.gameId]['teamELO'] = 1500
        else:
            dfs[dfs.gameId == fg.gameId]['oppTeamELO'] = 1500
        
    return dfs
        
    
    
    
    
        

if __name__ == '__main__':
    
    #read dataset    
    ts = pd.read_sql('teams_stats_by_game',engine,index_col='index')
    gs = pd.read_sql('games_schedule',engine)
    
    #add seasonId for convenience
    gs['seasonId'] = gs.apply(lambda x: int(str(x.gameId)[0:4]),axis=1)
    gs['preGameELO'] = -1
    gs['postGameELO'] = -1
    
    


#    #split in half, only home games    
#    df = df[df.gameLocationCode == 'H']
#    
#    
#    dfs = process_elo_forward(2010)
    
    dfs = gs[gs.seasonId == 2010]
    teamList = dfs.teamAbbrev.unique()
    
    #assign initial values
    for team in teamList:
        #first game of the season        
        teamGames = dfs[dfs.teamAbbrev == team].sort_values(by='gameDate')      
        fg = teamGames.iloc[0]
        dfs.preGameELO[(dfs.gameId == fg.gameId) & (dfs.teamAbbrev == team)]=1500