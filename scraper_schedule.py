# This file download the summary of the whole season from the following link:
# http://www.nhl.com/stats/rest/grouped/teams/game/teamsummary?cayenneExp=seasonId=20152016%20and%20gameTypeId=2
# It does not require the team names, just the season ID and game type ID

import json
import requests
import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine

#get schedule by dates
def get_schedule_by_dates(startDate,endDate):
    schedule = []
    hdrs = {
        'Host': 'statsapi.web.nhl.com',
        'Connection': 'keep-alive',
        'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-gb, en;q=0.7',
        'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    print('Getting schedule for: '+str(startDate)+' : '+str(endDate))

    url = 'http://statsapi.web.nhl.com/api/v1/schedule?startDate='
    url = url+ startDate + '&endDate=' + endDate 
    url = url+ '&expand=schedule.teams&leaderGameTypes=R&site=en_nhl'

    res = requests.get(url, headers = hdrs)
    data = json.loads(res.content.decode('utf-8'))

    dates = data['dates']

    for date in dates:
        games = date['games']
        for game in games:
            result = {}
            result['gameId'] = game['gamePk']
            result['teamIdHome'] = game['teams']['home']['team']['id']
            result['teamIdRoad'] = game['teams']['away']['team']['id']
            result['teamAbbrevHome'] = game['teams']['home']['team']['abbreviation']
            result['teamAbbrevRoad'] = game['teams']['away']['team']['abbreviation']
            result['gameDate'] = game['gameDate']
            schedule.append(result)

    schedule = pd.DataFrame(schedule)
    schedule['gameDate']=pd.to_datetime(schedule['gameDate'])
    return schedule

if __name__ == '__main__':
    sch = get_schedule_by_dates('2016-11-14','2016-11-15')


