import requests
import json
import time
import pandas as pd
from sqlalchemy import create_engine

hdrs = {
  'Host': 'www.nhl.com',
  'Connection': 'keep-alive',
  'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36',
  'Accept-Encoding': 'gzip, deflate, sdch',
  'Accept-Language': 'en-gb, en;q=0.7',
  'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
  }

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()
   
#make query for the provided dataframe of skaters and team names
def make_query_skaters(df,team,stats):
    players = df[df.playerTeamsPlayedFor.str.contains(team)]['playerId']
    query = 'http://www.nhl.com/stats/rest/skaters/'
    query = query + 'chart?cayenneExp=playerId%20in%20('
    for player in players:
        query = query + str(player) + ',%20'
    query = query[:-4]
    query = query+')%20and%20seasonId='+seasonId
    query = query+'%20and%20gameTypeId=2'
    for stat in stats:
        query = query+'&include='+stat.rstrip()
    query = query + '&sort=gameDate&dir=ASC'
    return query

#get all games for skaters
def get_all_games_skaters(season,roster,table):
    result = 0    
    
    players = pd.read_sql(roster,engine)
    players = players[players.seasonId == int(season)]
    players = players[['playerId','playerTeamsPlayedFor']]
    
    #get unique team list    
    teams = list(players.playerTeamsPlayedFor.unique())
    for team in teams:
        if len(team)>3:
            teams.extend(team.split(', '))
    teams = list(set([t for t in teams if len(t)==3]))
        
    #stats
    with open('stats_to_get_skaters.lst') as f:
        stats = f.readlines()

    #iterate over teams for skaters
    for team in teams:
        url = make_query_skaters(players,team.rstrip(),stats)
        print "parsing: "+team.rstrip()
        res = requests.get(url, headers = hdrs)
        data = json.loads(res.content)
        games = pd.DataFrame(data['data'])
        result += len(games)
        games.to_sql(table,engine,if_exists='append',index=False)
        time.sleep(10)
    return result
    
#get all games for goalies
def get_all_games_goalies(season,roster,table):
    
    players = pd.read_sql(roster,engine)
    players = players[players.seasonId == int(season)]
    goalies = players[players.playerPositionCode == 'G']
    goalies = goalies.playerId        
    
    #stats
    with open('stats_to_get_goalies.lst') as f:
        stats = f.readlines()
    
    #make query
    query = 'http://www.nhl.com/stats/rest/goalies/'
    query = query + 'chart?cayenneExp=playerId%20in%20('
    for goalie in goalies:
        query = query + str(goalie) + ',%20'
    query = query[:-4]
    query = query+')%20and%20seasonId='+seasonId
    query = query+'%20and%20gameTypeId=2'
    for stat in stats:
        query = query+'&include='+stat.rstrip()
    query = query + '&sort=gameDate&dir=ASC'

    res = requests.get(query, headers = hdrs)
    data = json.loads(res.content)
    games = pd.DataFrame(data['data'])
    games.to_sql(table,engine,if_exists='append',index=False)
    return len(goalies)

if __name__ == '__main__':
    
    seasonId = '20152016' #20052006 to 20152016
    roster = 'rosters'
    table = 'games'
    
    print get_all_games_skaters(seasonId,roster,table)
#    print get_all_games_goalies(seasonId,roster,table)
    print seasonId
    

#Player stats by game
#https://statsapi.web.nhl.com/api/v1/people/8475744/stats?stats=gameLog&expand=stats.team&season=20152016&site=en_nhl

#http://www.nhl.com/stats/rest/skaters/chart?
#cayenneExp=playerId%20in%20(8471685,%208475726,%208470604,%208474563,%208473473)
#%20and%20seasonId=20152016
#%20and%20gameTypeId=2
#&include=goals
#&include=assists
#&include=points
#&include=plusMinus
#&include=penaltyMinutes
#&include=ppGoals
#&include=shGoals
#&include=shPoints
#&include=gameWinningGoals
#&include=otGoals
#&include=shots
#&include=shootingPctg
#&include=timeOnIcePerGame
#&include=shiftsPerGame
#&include=faceoffWinPctg
#&include=teamAbbrev
#&include=opponentTeamAbbrev
#&include=gameId
#&include=teamId
#&include=gameDate
#&include=playerId
#&include=playerPositionCode
#&sort=gameDate&dir=ASC    