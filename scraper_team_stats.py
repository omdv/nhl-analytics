import json
import requests
import pandas as pd
import time
from sqlalchemy import create_engine

#initializing
hdrs = {
  'Host': 'www.nhl.com',
  'Connection': 'keep-alive',
  'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36',
  'Accept-Encoding': 'gzip, deflate, sdch',
  'Accept-Language': 'en-gb, en;q=0.7',
  'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
  }

#requests
with open('requests_teams.lst') as f:
    reqs = f.readlines()

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhl')
conn = engine.connect()

def get_df(reqtype,season,goalie):
    print 'Parsing: '+reqtype
    
    url = 'http://www.nhl.com/stats/rest/grouped/teams/'
    url = url + reqtype
    url = url + '?cayenneExp=seasonId='+season+'%20and%20gameTypeId=2'
    
    res = requests.get(url, headers = hdrs)
    data = json.loads(res.content)
    # return pd.DataFrame(data['data']).set_index('teamId')
    return pd.DataFrame(data['data'])

#get full dataset for a given season
def get_full_set(seasonId):
    teams = {}
    #get teams
    for req in reqs:
        url = req.rstrip()
        keys = url.split('/')
        key = keys[len(keys)-1]
        teams[key] = get_df(url,seasonId,False)
        time.sleep(10)
    
        final = pd.DataFrame()
        #combined dataframe for teams
        for key in teams:
            if final.empty:
                final = teams[key]
            else:
                df = teams[key]            
                cols_to_use = df.columns.difference(final.columns) 
                final = pd.merge(final, df[cols_to_use], left_index=True, 
                                 right_index=True, how='outer')
    return final


if __name__ == '__main__':
    season = '20092010'
    teams = get_full_set(season)

    #pickle teams
    teams.to_sql('teams',engine,if_exists='append')