import pandas as pd
import re

from datetime import datetime
from sqlalchemy import create_engine

pd.options.mode.chained_assignment = None  #suppress chained assignment warning

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()

#create schedule out of 'games' table
def parse_schedule(df,table):
    cgames = df[['gameDate','teamAbbrev','gameLocationCode','gameId']]
    cgames = cgames.drop_duplicates()
    cgames.to_sql(table, engine, if_exists='replace',index=False)
    return 0


#team abbreviations for a given game
def get_teams(df):
    game = games[games.gameId == df.gameId]
    df['homeTeam'] = game[game.gameLocationCode=='H'].teamAbbrev.values[0]
    df['roadTeam'] = game[game.gameLocationCode=='R'].teamAbbrev.values[0]
    return df

#get goals for a given game
#df.result - result for the home team
#df.homeTeamGoals (MT,OT,SH)
#df.roadTeamGoals (MT,OT,SH)
def get_goals(dfrow):
    game = games[games.gameId == dfrow.gameId]

    dfrow['homeTeamGoalsMT'] = game[game.gameLocationCode=='H'].goals.sum()
#    dfrow['homeTeamGoalsOT'] = game[game.gameLocationCode=='H'].otGoals.sum()
#    dfrow['homeTeamGoalsSH'] = game[game.gameLocationCode=='H'].shGoals.sum()
#    dfrow['homeTeamGoalsPP'] = game[game.gameLocationCode=='H'].ppGoals.sum()
    home = dfrow['homeTeamGoalsMT']
            
    dfrow['roadTeamGoalsMT'] = game[game.gameLocationCode=='R'].goals.sum()
#    dfrow['roadTeamGoalsOT'] = game[game.gameLocationCode=='R'].otGoals.sum()
#    dfrow['roadTeamGoalsSH'] = game[game.gameLocationCode=='R'].shGoals.sum()
#    dfrow['roadTeamGoalsPP'] = game[game.gameLocationCode=='R'].ppGoals.sum()
    road = dfrow['roadTeamGoalsMT']
            
    if home > road:
        dfrow['result'] = 'win'
    elif home < road:
        dfrow['result'] = 'loss'
    else: 
        dfrow['result'] = 'so'
    return dfrow

#get history for these two teams
#input - row of the df and the whole df
def get_history_opp(dfrow, df, period):
    hT = dfrow['homeTeam']
    rT = dfrow['roadTeam']
    now = dfrow['gameDate']
    
    #get two histories - at home and on the road
    hH = df[(df.homeTeam == hT) & (df.roadTeam == rT)]
    hH = hH[hH.gameDate < now].sort_values(by='gameDate')
    hR = df[(df.homeTeam == rT) & (df.roadTeam == hT)]
    hR = hR[hR.gameDate < now].sort_values(by='gameDate')
    
    dfrow['homeTeamHomeWinsVsOppPeriodMean'] = hH[-period:].win.mean()
    dfrow['homeTeamRoadWinsVsOppPeriodMean'] = hR[-period:].loss.mean()

    dfrow['homeTeamHomeLossVsOppPeriodMean'] = hH[-period:].loss.mean()
    dfrow['homeTeamRoadLossVsOppPeriodMean'] = hR[-period:].win.mean()
    
    dfrow['homeTeamHomeGoalsVsOppPeriodMean'] = hH[-period:].homeTeamGoalsMT.mean()
    dfrow['homeTeamRoadGoalsVsOppPeriodMean'] = hR[-period:].roadTeamGoalsMT.mean()

    dfrow['roadTeamHomeGoalsVsOppPeriodMean'] = hR[-period:].homeTeamGoalsMT.mean()
    dfrow['roadTeamRoadGoalsVsOppPeriodMean'] = hH[-period:].roadTeamGoalsMT.mean()
    
    dfrow['homeTeamHomeWinsVsOppAllMean'] = hH.win.mean()
    dfrow['homeTeamRoadWinsVsOppAllMean'] = hR.loss.mean()
    
    dfrow['homeTeamHomeLossVsOppAllMean'] = hH.loss.mean()
    dfrow['homeTeamRoadLossVsOppAllMean'] = hR.win.mean()
    
    dfrow['homeTeamHomeGoalsVsOppAllMean'] = hH.homeTeamGoalsMT.mean()
    dfrow['homeTeamRoadGoalsVsOppAllMean'] = hR.roadTeamGoalsMT.mean()

    dfrow['roadTeamHomeGoalsVsOppAllMean'] = hR.homeTeamGoalsMT.mean()
    dfrow['roadTeamRoadGoalsVsOppAllMean'] = hH.roadTeamGoalsMT.mean()
    return dfrow
    
#get history of the last 10 games
#input - row of the df and the whole df
def get_history_all(dfrow, df, period):
    hT = dfrow.homeTeam
    now = dfrow.gameDate
    
    #get two histories - at home and on the road
    hH = df[(df.homeTeam == hT) & 
            (df.gameDate < now)].sort_values(by='gameDate')
    hR = df[(df.roadTeam == hT) & 
            (df.gameDate < now)].sort_values(by='gameDate')
    
    dfrow['homeWinsVsAllPeriodMean'] = hH[-period:].win.mean()
    dfrow['roadWinsVsAllPeriodMean'] = hR[-period:].win.mean()
    
    dfrow['homeLossVsAllPeriodMean'] = hH[-period:].loss.mean()
    dfrow['roadLossVsAllPeriodMean'] = hR[-period:].loss.mean()
    
    dfrow['homeWinsVsAllMean'] = hH.win.mean()
    dfrow['roadWinsVsAllMean'] = hR.win.mean()
    
    dfrow['homeLossVsAllMean'] = hH.loss.mean()
    dfrow['roadLossVsAllMean'] = hR.loss.mean()
    
    dfrow['homeGoalsVsAllMean'] = hH.homeTeamGoalsMT.mean()
    dfrow['roadGoalsVsAllMean'] = hR.roadTeamGoalsMT.mean()
    return dfrow
  
#get features out of the matrix of previous results vs all teams
def get_history_matrix(dfrow,games):
    teams = list(games.homeTeam.unique())
    
    #all history
    hA = games[(games.homeTeam == dfrow.homeTeam) |
            (games.roadTeam == dfrow.homeTeam) &
            (games.gameDate < dfrow.gameDate)].sort_values(by='gameDate')
    
#    for team in teams:
#        dfrow[team] = 
    
    return df


#create a set describing roster strength
#input - training set row
#input - players dataset
def get_roster_features(dfrow):
    game = games.loc[dfrow.gameId]
    pgames = games[games.gameDate < dfrow.gameDate]
    
    #get rosters for a given game
    homePl = game[game.gameLocationCode == 'H'] \
        [['playerId','playerPositionCode']]
    roadPl = game[game.gameLocationCode == 'R'] \
        [['playerId','playerPositionCode']]


    #all history for home team and road team
    homeTmH = pgames[pgames.teamAbbrev == dfrow.homeTeam]
    roadTmH = pgames[pgames.teamAbbrev == dfrow.roadTeam]

    hGoalie = homePl[homePl.playerPositionCode == 'G'].playerId.values
    rGoalie = roadPl[roadPl.playerPositionCode == 'G'].playerId.values
    
    hDef = homePl[homePl.playerPositionCode == 'D'].playerId.values
    rDef = roadPl[roadPl.playerPositionCode == 'D'].playerId.values
    
    hWng = homePl[homePl.playerPositionCode.str.contains('R|L')].playerId.values
    rWng = roadPl[roadPl.playerPositionCode.str.contains('R|L')].playerId.values
    
    hCnt = homePl[homePl.playerPositionCode == 'C'].playerId.values
    rCnt = roadPl[roadPl.playerPositionCode == 'C'].playerId.values
    
#    print hGoalie
#    print rGoalie
#    
##    print hGoalie, rGoalie
##    print hDef, rDef
##    print hWng, rWng

    #goalie features
#    dfrow['homeGoalieSavePctgAllMean'] = \
#            homeTmH[homeTmH.playerId.isin(hGoalie)].savePctg.mean()
#    dfrow['homeGoalieSavePctgLastMean'] = \
#            homeTmH[homeTmH.playerId.isin(hGoalie)][-1:].savePctg.mean()
#            
#    dfrow['roadGoalieSavePctgAllMean'] = \
#            roadTmH[roadTmH.playerId.isin(rGoalie)].savePctg.mean()
#    dfrow['roadGoalieSavePctgLastMean'] = \
#            roadTmH[roadTmH.playerId.isin(rGoalie)][-1:].savePctg.mean()

    dfrow['homeGoalieShotsAgainstAllMean'] = \
            homeTmH[homeTmH.playerId.isin(hGoalie)].shotsAgainst.mean()
    dfrow['homeGoalieShotsAgainstLastMean'] = \
            homeTmH[homeTmH.playerId.isin(hGoalie)][-1:].shotsAgainst.mean()
            
    dfrow['roadGoalieShotsAgainstAllMean'] = \
            roadTmH[roadTmH.playerId.isin(rGoalie)].shotsAgainst.mean()
    dfrow['roadGoalieShotsAgainstLastMean'] = \
            roadTmH[roadTmH.playerId.isin(rGoalie)][-1:].shotsAgainst.mean()
    
    #defense features
#    dfrow['homeDefPlusMinusAllMean'] = \
#            homeTmH[homeTmH.playerId.isin(hDef)].plusMinus.mean()
#    dfrow['homeDefPlusMinusLastMean'] = \
#            homeTmH[homeTmH.playerId.isin(hDef)][-1:].plusMinus.mean()
#    
#    dfrow['roadDefPlusMinusAllMean'] = \
#            roadTmH[roadTmH.playerId.isin(rDef)].plusMinus.mean()
#    dfrow['roadDefPlusMinusLastMean'] = \
#            roadTmH[roadTmH.playerId.isin(rDef)][-1:].plusMinus.mean()
    
    dfrow['homeDefShotsAllMean'] = \
            homeTmH[homeTmH.playerId.isin(hDef)].shots.mean()
    dfrow['homeDefShotsLastMean'] = \
            homeTmH[homeTmH.playerId.isin(hDef)][-1:].shots.mean()
    
    dfrow['roadDefShotsAllMean'] = \
            roadTmH[roadTmH.playerId.isin(rDef)].shots.mean()
    dfrow['roadDefShotsLastMean'] = \
            roadTmH[roadTmH.playerId.isin(rDef)][-1:].shots.mean()
        
    #winger features
#    dfrow['homeWngPlusMinusAllMean'] = \
#            homeTmH[homeTmH.playerId.isin(hWng)].plusMinus.mean()
#    dfrow['homeWngPlusMinusLastMean'] = \
#            homeTmH[homeTmH.playerId.isin(hWng)][-1:].plusMinus.mean()
#    
#    dfrow['roadWngPlusMinusAllMean'] = \
#            roadTmH[roadTmH.playerId.isin(rWng)].plusMinus.mean()
#    dfrow['roadWngPlusMinusLastMean'] = \
#            roadTmH[roadTmH.playerId.isin(rWng)][-1:].plusMinus.mean()
        
    dfrow['homeWngShotsAllMean'] = \
            homeTmH[homeTmH.playerId.isin(hWng)].shots.mean()
    dfrow['homeWngShotsLastMean'] = \
            homeTmH[homeTmH.playerId.isin(hWng)][-1:].shots.mean()
    
    dfrow['roadWngShotsAllMean'] = \
            roadTmH[roadTmH.playerId.isin(rWng)].shots.mean()
    dfrow['roadWngShotsLastMean'] = \
            roadTmH[roadTmH.playerId.isin(rWng)][-1:].shots.mean()
            
    #center features
#    dfrow['homeCntPlusMinusAllMean'] = \
#            homeTmH[homeTmH.playerId.isin(hCnt)].plusMinus.mean()
#    dfrow['homeCntPlusMinusLastMean'] = \
#            homeTmH[homeTmH.playerId.isin(hCnt)][-1:].plusMinus.mean()
#    
#    dfrow['roadCntPlusMinusAllMean'] = \
#            roadTmH[roadTmH.playerId.isin(rCnt)].plusMinus.mean()
#    dfrow['roadCntPlusMinusLastMean'] = \
#            roadTmH[roadTmH.playerId.isin(rCnt)][-1:].plusMinus.mean()
            
    dfrow['homeCntShotsAllMean'] = \
            homeTmH[homeTmH.playerId.isin(hCnt)].shots.mean()
    dfrow['homeCntShotsLastMean'] = \
            homeTmH[homeTmH.playerId.isin(hCnt)][-1:].shots.mean()
    
    dfrow['roadCntShotsAllMean'] = \
            roadTmH[roadTmH.playerId.isin(rCnt)].shots.mean()
    dfrow['roadCntShotsLastMean'] = \
            roadTmH[roadTmH.playerId.isin(rCnt)][-1:].shots.mean()

    return dfrow


if __name__ == '__main__':
    table_game = 'games'
    table_sche = 'schedule'
    
    #read dataset    
#    games = pd.read_sql(table_game,engine).set_index('gameId')
#    schedule = pd.read_sql(table_sche,engine)
#    training = pd.read_sql('training',engine)
    
    #create schedule table
#    parse_schedule(games,table_sche)
    
    #feature engineering
    
    #Step 1
#    training = schedule[['gameId','gameDate']]
#    training.drop_duplicates(inplace=True)
    
    #Step 2: parse datetime
#    pat = re.compile('T|Z')
#    fmt = '%Y-%m-%d %H:%M:%S'
#    training['gameDate'] = training.apply(lambda row: 
#        datetime.strptime(pat.sub(' ',row['gameDate'])[:-1],fmt),axis=1)
    
    #Step 3: parse_teams    
#    training = training.apply(get_teams, axis=1)
    
    #Step 4: get result and dummies
    #result for the game and create dummies
#    training = training.apply(get_goals, axis=1)
#    training = pd.concat([training, pd.get_dummies(training.result)], axis=1)
    
    #Step 5: history of games vs opponent
#    training = training.apply(lambda row: 
#        get_history_opp(row, training, 3), axis=1)
    
    #Step 6: history of all last games
#    training = training.apply(lambda row: get_history_all(row, training, 10), axis=1)
    
    #Step 7: roster features
    training = training.apply(get_roster_features, axis=1)
    
    #Pickle
#    training.to_sql('training',engine, if_exists='replace', index=False)    