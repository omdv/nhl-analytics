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
with open('requests.lst') as f:
    reqs = f.readlines()

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()

def get_df(reqtype,season,goalie):
    print 'Parsing: '+reqtype
    
    if not goalie:
        url = 'http://www.nhl.com/stats/rest/grouped/skaters/'
        url = url + reqtype
        url = url + '?cayenneExp=seasonId='+season+'%20and%20gameTypeId=2'
    if goalie:
        url = 'http://www.nhl.com/stats/rest/grouped/goalies/'
        url = url + 'season/goaliesummary'
        url = url + '?cayenneExp=seasonId='+season
        url = url + '%20and%20gameTypeId=2%20and%20playerPositionCode=%22G%22'
    
    res = requests.get(url, headers = hdrs)
    data = json.loads(res.content)
    return pd.DataFrame(data['data']).set_index('playerId')

#get roster only
def get_roster(seasonId):
    goalies = get_df('season/goaliesummary',seasonId,True)
    time.sleep(10)    
    skaters = get_df('season/skatersummary',seasonId,False)
    return skaters,goalies

#get full dataset for a given season
def get_full_set(seasonId):
    #get goalies
    goalies = get_df('season/goaliesummary',seasonId,True)
    time.sleep(10)
    
    skaters = {}
    #get skaters
    for req in reqs:
        url = req.rstrip()
        keys = url.split('/')
        key = keys[len(keys)-1]
        skaters[key] = get_df(url,seasonId,False)
        time.sleep(10)
    
        final = pd.DataFrame()
        #combined dataframe for skaters
        for key in skaters:
            if final.empty:
                final = skaters[key]
            else:
                df = skaters[key]            
                cols_to_use = df.columns.difference(final.columns) 
                final = pd.merge(final, df[cols_to_use], left_index=True, 
                                 right_index=True, how='outer')
    return final,goalies


if __name__ == '__main__':
    season = '20052006'
    skaters, goalies = get_roster(season)

    #pickle skaters
    skaters.to_sql('rosters',engine,if_exists='append')
       
    #pickle goalies
#    cols_to_add = goalies.columns.difference(skaters.columns)
#    for col in cols_to_add:
#        result = conn.execute("ALTER TABLE public.rosters ADD COLUMN \""+ col +"\" bigint;")
    goalies.to_sql('rosters',engine,if_exists='append')
