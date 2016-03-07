import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

#classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from datetime import datetime
from sqlalchemy import create_engine

pd.options.mode.chained_assignment = None  #suppress chained assignment warning

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()

#get schedule out of the list of the games
#need to be run only when parsing new
def parse_schedule(df,table):
    result = 0
    games = list(df.gameId.unique())
    
    for game in games:
        cgame = df[df.gameId == int(game)]
        cgame = cgame[['gameDate','teamAbbrev','gameLocationCode','gameId']]
        cgame = cgame.drop_duplicates()
        result += len(cgame)
        cgame.to_sql('schedule',engine, if_exists='append',index=False)
    return result


#team abbreviations for a given game
def get_team(dfrow,games,code):
    game = games[games.gameId == dfrow.gameId]
    team = game[game.gameLocationCode== code].teamAbbrev.unique()[0]
    return team

#goals for a given game
def get_goals(dfrow,games):
    game = games[games.gameId == dfrow.gameId]
    homeMT = game[game.gameLocationCode=='H'].goals.sum()
    homeOT = game[game.gameLocationCode=='H'].otGoals.sum()
    homeSH = game[game.gameLocationCode=='H'].shGoals.sum()
    home = homeMT + homeOT + homeSH
    roadMT = game[game.gameLocationCode=='R'].goals.sum()
    roadOT = game[game.gameLocationCode=='R'].otGoals.sum()
    roadSH = game[game.gameLocationCode=='R'].shGoals.sum()
    road = roadMT + roadOT + roadSH
    if home > road:
        result = 1
    elif home < road:
        result = 0
    else: 
        result = 0.5
#    win = (True if home > road else False)
    return result

#get history for these two games
#input - row of the df and the whole df
def get_history(dfrow, df):
    hT = dfrow['homeTeam']
    rT = dfrow['roadTeam']
    now = dfrow['gameDate']
#    print hT + ' vs ' + rT
    
    #get two histories - at home and on the road
    hH = df[(df.homeTeam == hT) & (df.roadTeam == rT)]
    hH = hH[hH.gameDate < now]
    hR = df[(df.homeTeam == rT) & (df.roadTeam == hT)]
    hR = hR[hR.gameDate < now]
    
    #how many times did the home team win at home before
    if len(hH) == 0:
        winHome = 0
    elif len(hH) >= 5:
        winHome = hH[-5:].win.sum()/5
    else:
        winHome = hH.win.sum()/len(hH)
    
    #how many times did the home team win on the road before
    if len(hR) == 0:
        winRoad = 0
    elif len(hR) >= 5:
        winRoad = 1 - hR[-5:].win.sum()/5
    else:
        winRoad = 1 - hR.win.sum()/len(hR)   
        
    return [winHome,winRoad]

#create a dataset
#input - schedule and games dataframe
def create_dataset(sch,g):
    df = sch[['gameId','gameDate']]
    df.drop_duplicates(inplace=True)
    
    df['win'] = df.apply(lambda row: get_goals(row, g), axis=1)
    df['homeTeam'] = df.apply(lambda row: get_team(row, g,'H'), axis=1)
    df['roadTeam'] = df.apply(lambda row: get_team(row, g,'R'), axis=1)
    
    #parse datetime
    pat = re.compile('T|Z')
    fmt = '%Y-%m-%d %H:%M:%S'
    df['gameDate'] = df.apply(lambda row: 
        datetime.strptime(pat.sub(' ',row['gameDate'])[:-1],fmt),axis=1)
    
    #home and road wins percentage for the current home team
    df['homeWins'] = df.apply(lambda row: get_history(row, df)[0], axis=1)
    df['roadWins'] = df.apply(lambda row: get_history(row, df)[1], axis=1)
    
    return df
    

if __name__ == '__main__':
    table_game = 'games'
    table_sche = 'schedule'
    
    #read dataset    
#    games = pd.read_sql(table_game,engine)
#    schedule = pd.read_sql(table_sche,engine)
    
    #create schedule out of rawgames - need only periodically
#    print parse_schedule(games,tableou)
    
    #feature engineering
#    full = create_dataset(schedule,games)
    
#    #shuffle dataset
#    fulltrim = full[full.gameId > 2012020000]
#    fulltrim = fulltrim.apply(np.random.permutation)
#    
#    #split in sets
#    train = fulltrim[:int(len(fulltrim)*0.6)]
#    val = fulltrim[int(len(fulltrim)*0.6):int(len(fulltrim)*0.8)]
#    test = fulltrim[int(len(fulltrim)*0.8):len(fulltrim)]
#    
    X_train = train[['homeWins','roadWins']].values.tolist()
##    X_train = np.asarray(train[['homeWins','roadWins']], dtype="|S6")
    Y_train = np.asarray(train.win, dtype="|S6")
    
    #logistic regression
#    logistic = LogisticRegression(max_iter=500,tol=0.001)
#    logistic.fit(X_train,Y_train)
    
#    forest = RandomForestClassifier(oob_score=True, n_estimators=50)
#    forest.fit(X_train,Y_train)
#    feature_importance = forest.feature_importances_

#    yval = forest.predict(val[['homeWins','roadWins']])    
#    ytest = forest.predict(test[['homeWins','roadWins']])    
