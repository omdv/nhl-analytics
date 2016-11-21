# This file download the summary of the whole season from the following link:
# http://www.nhl.com/stats/rest/grouped/teams/game/teamsummary?cayenneExp=seasonId=20152016%20and%20gameTypeId=2
# It does not require the team names, just the season ID and game type ID

import json
import requests
import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine

#preparing headers
hdrs = {
  'Host': 'www.nhl.com',
  'Connection': 'keep-alive',
  'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36',
  'Accept-Encoding': 'gzip, deflate, sdch',
  'Accept-Language': 'en-gb, en;q=0.7',
  'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
  }

#preparing postgres engine
engine = create_engine('postgresql://postgres:@192.168.99.100:5432/nhlstats')
conn = engine.connect()


#get team summary by season by game type
def get_team_summary_by_season(season):
    print('Parsing: '+str(season))
    
    url = 'http://www.nhl.com/stats/rest/grouped/teams/game/'
    url = url + 'teamsummary?cayenneExp=seasonId='+season
    
    res = requests.get(url, headers = hdrs)
    data = json.loads(res.content.decode('utf-8'))
    return pd.DataFrame(data['data'])

def save_team_summary_to_sql(season):
    df = get_team_summary_by_season(season)

    # add season ID and gameTypeId
    df['seasonId'] = df.apply(lambda x: int(str(x.gameId)[0:4]),axis=1)
    df['gameType'] = df.apply(lambda x: int(str(x.gameId)[5:6]),axis=1)
    df = df[df.gameType.isin([2,3])]

    del df['ties']
    del df['gamesPlayed']

    # get if OT was played
    df['otPlayed'] = np.logical_or(df.otLossesHome,df.otLossesRoad).astype(np.int8)

    # remove correlated features
    for key in ['gameLocationCodeHome',
    'gameLocationCodeRoad',
    'opponentTeamAbbrevHome',
    'opponentTeamAbbrevRoad',
    'faceoffWinPctgRoad',
    'faceoffsLostRoad',
    'faceoffsWonRoad',
    'shotsForRoad',
    'shotsAgainstRoad',
    'goalsForRoad',
    'goalsAgainstRoad',
    'ppGoalsAgainstRoad',
    'ppGoalsForRoad',
    'gameTypeRoad',
    'seasonIdRoad',
    'gameDateRoad',
    'lossesHome',
    'lossesRoad',
    'winsRoad',
    'otLossesHome',
    'otLossesRoad',
    'ppOpportunitiesRoad',
    'shNumTimesRoad',
    'pointsHome',
    'pointsRoad']:
      del df[key]

    df.rename(\
      columns={
        'gameDateHome':'gameDate',
        'seasonIdHome':'seasonId',
        'gameTypeHome':'gameType'},\
      inplace=True)

    # merge with existing
    df_old = pd.read_sql('team_stats_by_game',engine)
    df = pd.concat([df_old,df])
    df.drop_duplicates(inplace=True)
    
    df.set_index('gameId',inplace=True)

    df.to_sql('team_stats_by_game',engine,if_exists='replace')
    return 0


if __name__ == '__main__':
    seasonId = ['20062007','20072008','20082009','20092010','20102011',\
    '20112012','20122013','20132014','20142015','20152016','20162017']

#SCRAPER PART
    # mapping first season
    df = get_team_summary_by_season('20052006')

    # iterate over seasons
    for season in seasonId:
      time.sleep(10)
      df = pd.concat([df,get_team_summary_by_season(season)])

#PROCESSING PART
    df.drop_duplicates(inplace=True)

    # add season ID and gameTypeId
    df['seasonId'] = df.apply(lambda x: int(str(x.gameId)[0:4]),axis=1)
    df['gameType'] = df.apply(lambda x: int(str(x.gameId)[5:6]),axis=1)
    df = df[df.gameType.isin([2,3])]

    # df.gameType = df.gameType.astype('category')
    # df.gameType = df.gameType.cat.rename_categories(['R'])

    del df['ties']
    del df['gamesPlayed']
    df.sort_values(by='gameId',inplace=True)

    # dfh = df[df.gameLocationCode=='H']
    # dfr = df[df.gameLocationCode=='R']

    # dfh.columns = [x+'Home' for x in dfh.columns]
    # dfr.columns = [x+'Road' for x in dfr.columns]
    # dfh.rename(columns={'gameIdHome':'gameId'},inplace=True)
    # dfr.rename(columns={'gameIdRoad':'gameId'},inplace=True)

    # df = pd.merge(dfh,dfr,on='gameId',how='left')

    # get if OT was played
    # df['otPlayed'] = np.logical_or(df.otLosses,df.otLossesRoad).astype(np.int8)

    # # remove correlated features
    # for key in ['gameLocationCodeHome',
    # 'gameLocationCodeRoad',
    # 'opponentTeamAbbrevHome',
    # 'opponentTeamAbbrevRoad',
    # 'faceoffWinPctgRoad',
    # 'faceoffsLostRoad',
    # 'faceoffsWonRoad',
    # 'shotsForRoad',
    # 'shotsAgainstRoad',
    # 'goalsForRoad',
    # 'goalsAgainstRoad',
    # 'ppGoalsAgainstRoad',
    # 'ppGoalsForRoad',
    # 'gameTypeRoad',
    # 'seasonIdRoad',
    # 'gameDateRoad',
    # 'lossesHome',
    # 'lossesRoad',
    # 'winsRoad',
    # 'otLossesHome',
    # 'otLossesRoad',
    # 'ppOpportunitiesRoad',
    # 'shNumTimesRoad',
    # 'pointsHome',
    # 'pointsRoad']:
    #   del df[key]

    # df.rename(\
    #   columns={
    #     'gameDateHome':'gameDate',
    #     'seasonIdHome':'seasonId',
    #     'gameTypeHome':'gameType'},\
    #   inplace=True)
    # df = df.set_index('gameId')

    # merge with existing
    # df_old = pd.read_sql('team_stats_by_game',engine)
    # df = pd.concat([df_old,df])
    # df.drop_duplicates(inplace=True)
    
    df.set_index('gameId',inplace=True)

    df.to_sql('team_stats_by_game',engine,if_exists='replace')


