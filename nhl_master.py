import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import trueskill as ts
import xgboost as xgb
import datetime as dt
from sqlalchemy import create_engine

np.random.seed(42)
pd.options.display.float_format = '{:.2f}'.format

#preparing postgres engine
engine = create_engine('postgresql://postgres:@192.168.99.100:5432/nhlstats')

def read_dataset_from_sql():
    print("Load dataset...")
    df = pd.read_sql('team_stats_by_game',engine,parse_dates=['gameDate'])

    df['dtindex'] = pd.DatetimeIndex(df.gameDate)
    startDate = df[df.gameId==2005020001].dtindex
    df['timeShift'] = df.apply(lambda row: (row.dtindex-startDate), axis=1)
    df['timeShift'] = df.apply(lambda row: row['timeShift'].days, axis=1)

    print('Process dataset...')
    df = df.sort_values(by='gameId')
    return df

if __name__ == '__main__':


    # get_team_summary_by_season('20162017')
    df = read_dataset_from_sql()
    
    # # --------
    # nYears = save_for_infernet(df)
    # call("mono /Users/om/Documents/Visual\ Studio\ 2015/Projects/TrueSkillTT/TrueSkillTT/bin/Release/TrueSkillTT.exe "+str(nYears),shell=True)
    # df,means,stds = merge_with_infernet(df)
    # # --------

    # # # merge with schedule
    # today = dt.date.today().strftime('%Y-%m-%d')
    # sch = get_schedule_by_dates(today,today)
    # df = merge_with_schedule(df,sch)

    # # pivot means and std by team
    # print("Pivot infer.net output by team...")
    # meansByTeam = means.pivot(index='timeShift',columns='teamId',values='tsMean')
    # stdsByTeam = stds.pivot(index='timeShift',columns='teamId',values='tsMean')

    # # shift skill by team forward, use last value
    # # TODO: this needs to be run daily, otherwise the gap between current time and last value will be too big
    # print("Assigning last value of the skill...")
    # missingGames = df[df.winsHome.isnull()].gameId.unique()
    # for g in missingGames:
    #     teamHome = df.ix[df.gameId == g,'teamIdHome'].values[0]
    #     teamRoad = df.ix[df.gameId == g,'teamIdRoad'].values[0]
    #     df.ix[df.gameId == g,'teamIdHomeMean'] = meansByTeam.ix[meansByTeam.index.max(),teamHome]
    #     df.ix[df.gameId == g,'teamIdRoadMean'] = meansByTeam.ix[meansByTeam.index.max(),teamRoad]
    #     df.ix[df.gameId == g,'teamIdHomeStd'] = stdsByTeam.ix[stdsByTeam.index.max(),teamHome]
    #     df.ix[df.gameId == g,'teamIdRoadStd'] = stdsByTeam.ix[stdsByTeam.index.max(),teamRoad]

    # # df['winProb'] = df.apply(lambda row: win_prob_2(row,131, 127.4),axis=1)
    # # df = get_rest_days(df)

    # features=[
    #     'homeMean',
    #     'homeStd',
    #     'roadMean',
    #     'roadStd',
    #     'restDaysHome',
    #     'restDaysRoad'
    # ]
    
    # gbm,imp,true,pred = run_single_xgboost(df[df.seasonId>2013],features,"winsHome",10,100)
    # threshold = get_optimal_cutoff(true,pred)
    # get_evaluation(true,pred,threshold)
    
    # # assign probability and result based on threshold
    # df['gamePredProb'] = pred
    # df['gamePredWins'] = np.where(pred>ths,1,0)



