import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pystan as ps
import pandas as pd
import numpy as np

np.random.seed(42)

def read_dataset():
    print("Load csv...")
    df = pd.read_csv('team_stats_by_game.csv',parse_dates=['gameDate'])
    df['week'] = pd.DatetimeIndex(df.gameDate)
    df['week'] = df.apply(lambda row: row.week.week+(row.week.year-2005)*52,axis=1)

    print('Process dataset...')
    df['gameType'],fct = pd.factorize(df.gameType)
    df = df.set_index('gameId').sort_values(by='gameDate')
    return df

if __name__ == '__main__':

    code = """
    data {
        int<lower=1> nGames; // number of games
        int<lower=1> nTeams; //number of teams
        real winsHome[nGames]; // result of the game
        int teamIdHome[nGames]; // id of home team
        int teamIdRoad[nGames]; // id of road team
    }
    parameters {
        real<lower=0,upper=300> teamMean[nTeams];
        real<lower=0,upper=3000> teamStd[nTeams];
        real<lower=0> teamSkill[nTeams];
        real<lower=0> performancePrecision;
        // real<lower=0,upper=300> homeAdvMean;
        // real<lower=0,upper=3000> homeAdvStd;
        // real<lower=0> homeAdv;
    }
    transformed parameters {
    }
    model {
        teamSkill ~ normal(teamMean,teamStd);
        // homeAdv ~ normal(homeAdvMean,homeAdvStd);
        for (g in 1:nGames)
            winsHome[g] ~ normal(teamSkill[teamIdHome[g]]-teamSkill[teamIdRoad[g]],performancePrecision);
    }
    """

    df = read_dataset()
    df = df[df.seasonId == 2015]
    winsHome = (df.winsHome.mean()-df.winsHome).values
    df['teamIdHome'], labels = pd.factorize(df['teamIdHome'])
    df['teamIdRoad'] = df.apply(lambda row: [i for i,x in enumerate(labels) if x == row.teamIdRoad],axis=1)

    teamIdHome = (df.teamIdHome+1).values
    teamIdRoad = (df.teamIdRoad+1).values
    nTeams = df.teamIdHome.unique().max()+1
    nGames = df.shape[0]


    data = {
        'nGames': nGames,
        'nTeams': nTeams,
        'winsHome': winsHome,
        'teamIdHome': teamIdHome,
        'teamIdRoad': teamIdRoad
        }

    print("Starting stan...")
    fit = ps.stan(model_name='pairwise_ranking',model_code=code,data=data,iter=1000, chains=4, n_jobs=-1)
