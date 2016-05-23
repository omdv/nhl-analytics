# This file download the summary of the whole season from the following link:
# http://www.nhl.com/stats/rest/grouped/teams/game/teamsummary?cayenneExp=seasonId=20152016%20and%20gameTypeId=2
# It does not require the team names, just the season ID and game type ID

import json
import requests
import pandas as pd
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
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhl')
conn = engine.connect()


#get team summary by season by game type
def get_team_summary_by_season(season,gameTypeId):
    print 'Parsing: '+str(season)
    
    url = 'http://www.nhl.com/stats/rest/grouped/teams/game/'
    url = url + 'teamsummary?cayenneExp=seasonId='
    url = url+season+'%20and%20gameTypeId='+gameTypeId
    
    res = requests.get(url, headers = hdrs)
    data = json.loads(res.content)
    return pd.DataFrame(data['data']).set_index(['gameId'])


if __name__ == '__main__':
    seasonId = ['20062007','20072008','20082009','20092010','20102011',\
    '20112012','20122013','20132014','20142015']

#SCRAPER PART
    # mapping first season
    # df = pd.concat([get_team_summary_by_season('20052006','2'),get_team_summary_by_season('20052006','3')])

    # iterate over seasons
    # for season in seasonId:
    #   time.sleep(10)
    #   df = pd.concat([df,get_team_summary_by_season(season,'2')])
    #   time.sleep(10)
    #   df = pd.concat([df,get_team_summary_by_season(season,'3')])

    # scrapping last season
    # df = pd.concat([df,get_team_summary_by_season('20152016','2')])
    # df = pd.concat([df,get_team_summary_by_season('20152016','3')])

    # writing to database
    # df.to_sql('team_stats_by_game',engine,if_exists='append')

#PROCESSING PART
    # read existing dataframe
    df = pd.read_sql('team_stats_by_game',engine)

    # add season ID and gameTypeId
    df['seasonId'] = df.apply(lambda x: int(str(x.gameId)[0:4]),axis=1)
    df['gameTypeId'] = df.apply(lambda x: int(str(x.gameId)[5:6]),axis=1)

    # rewrite the dataframe
    df.to_sql('team_stats_by_game',engine,if_exists='replace')
