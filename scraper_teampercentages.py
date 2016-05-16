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

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhl')
conn = engine.connect()


def get_df(season,team):
    print 'Parsing: '+str(team)
    
    url = 'http://www.nhl.com/stats/rest/grouped/teams/enhanced/game/'
    url = url+'teampercentages?cayenneExp=seasonId='
    url = url+season+'%20and%20gameTypeId=2%20and%20teamId='+str(team)
    
    res = requests.get(url, headers = hdrs)
    data = json.loads(res.content)
    return pd.DataFrame(data['data'])


if __name__ == '__main__':
    season = '20132014'

    #requests
    sql = 'SELECT DISTINCT teams."teamId" FROM public.teams WHERE teams."seasonId" = 20132014'
    result = conn.execute(sql)
    for row in result:
        team = int(row[0])
        teampercentages = get_df(season,team)
        teampercentages.to_sql('teampercentages',engine,if_exists='append')
        time.sleep(10)