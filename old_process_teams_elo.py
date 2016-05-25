import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.stats import t
from scipy.stats import norm

pd.options.mode.chained_assignment = None  #suppress chained assignment warning
pd.options.display.float_format = '{:.2f}'.format

#postgres
engine = create_engine(
    'postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()

#ELO params
ELO_DELTA = 200.
ELO_MEAN = 1500.
ELO_HOMEBONUS = 45
ELO_K_FACTOR = 16
ELO_OTWIN_COEFF = 0.75


#based on http://central.scipy.org/item/50/1/line-fit-with-confidence-intervals
def fitLine(x, y, alpha=0.05, plotFlag=1):
    # Summary data
    n = len(x)
    
    Sxx = np.sum(x**2) - np.sum(x)**2/n
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Linefit
    b = Sxy/Sxx
    a = mean_y - b*mean_x
    
    # Residuals
    fit = lambda xx: a + b*xx    
    residuals = y - fit(x)
    
    var_res = np.sum(residuals**2)/(n-2)
    sd_res = np.sqrt(var_res)
    
    # OM - added R2
    sst = np.sum((y-mean_y)**2)
    ssr = np.sum((fit(x)-mean_y)**2)
    r2 = ssr/sst
    
    # OM - for residuals plot
    res_mean = np.mean(residuals)
    res_var = np.sum((residuals - res_mean)**2)/n
    res_std = np.sqrt(res_var)
    
    # Confidence intervals
    se_b = sd_res/np.sqrt(Sxx)
    se_a = sd_res*np.sqrt(np.sum(x**2)/(n*Sxx))
    
    df = n-2                      # degrees of freedom
    tval = t.isf(alpha/2., df) 	# appropriate t value
    
    ci_a = a + tval*se_a*np.array([-1,1])
    ci_b = b + tval*se_b*np.array([-1,1])

    # create series of new test x-values to predict for
    npts = 100
    px = np.linspace(np.min(x),np.max(x),num=npts)
    
    se_fit     = lambda x: sd_res * np.sqrt(  1./n + (x-mean_x)**2/Sxx)
    se_predict = lambda x: sd_res * np.sqrt(1+1./n + (x-mean_x)**2/Sxx)
    
    print 'Summary: a={0:5.4f}+/-{1:5.4f}, b={2:5.4f}+/-{3:5.4f}, r2={4:6.4f}'.format(a,tval*se_a,b,tval*se_b,r2)
    print 'Confidence intervals: ci_a=({0:5.4f} - {1:5.4f}), ci_b=({2:5.4f} - {3:5.4f})'.format(ci_a[0], ci_a[1], ci_b[0], ci_b[1])
    print 'Residuals: variance = {0:5.4f}, standard deviation = {1:5.4f}'.format(var_res, sd_res)
    print 'alpha = {0:.3f}, tval = {1:5.4f}, df={2:d}'.format(alpha, tval, df)
    
    # Return info
    ri = {'residuals': residuals, 
        'var_res': var_res,
        'sd_res': sd_res,
        'alpha': alpha,
        'tval': tval,
        'df': df,
        'r2': r2}
    
    if plotFlag == 1:
        # Plot the data
        plt.figure(figsize=(14,8))
        
        plt.plot(px, fit(px),'k', label='Regression line')
        plt.plot(x,y,'r.', label='Sample observations')
        
#        x.sort()
        limit = (1-alpha)*100
        plt.plot(x, fit(x)+tval*se_fit(x), 'r--', label='Confidence limit ({0:.1f}%)'.format(limit))
        plt.plot(x, fit(x)-tval*se_fit(x), 'r--')
        
        plt.plot(x, fit(x)+tval*se_predict(x), 'c--', label='Prediction limit ({0:.1f}%)'.format(limit))
        plt.plot(x, fit(x)-tval*se_predict(x), 'c--')

        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Linear regression and confidence limits (R2: {:05.4f})'.format(r2))
        
        # configure legend
        plt.legend(loc=0)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=10)
        plt.show()
        
        
        # Plot the residuals
        plt.figure(figsize=(14,8))
        
        plt.plot(x,(y-x),'r.',label='Residuals')
        plt.xlabel('X values')
        plt.ylabel('Absolute residuals')
        plt.title('Residuals')
        
        res_ci_lo = norm.ppf(0.05, loc=res_mean, scale=res_std)
        res_ci_up = norm.ppf(0.95, loc=res_mean, scale=res_std)
        plt.plot([np.min(x),np.max(x)],[res_ci_lo,res_ci_lo], 'b--', label='90% range')
        plt.plot([np.min(x),np.max(x)],[res_ci_up,res_ci_up], 'b--')
        
        # configure legend
        plt.legend(loc=0)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=10)
        plt.show()
        
        # Plot the residuals histogram
        plt.figure(figsize=(14,8))
        
        n, bins, patches = plt.hist(residuals,50,normed=1,label='Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Distribution')
        plt.title('Residuals histogram')
        
        y = mlab.normpdf(bins, res_mean, res_std)
        plt.plot(bins, y, 'r--', linewidth=1,label='Normal')
        
        # configure legend
        plt.legend(loc=0)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=10)
        plt.show()        
        
        return ri

def process_elo_forward(season):
    gs['preGameELO'] = -1.0
    gs['postGameELO'] = -1.0
    gs['gamePredict'] = -1.0
    gs['gameResult'] = -1.0

    dfs = gs[gs.seasonId == 2010].sort_values(by='gameId')
    
    #loop over games - after init
    gameList = dfs.gameId.unique()
    teamList = dfs.teamAbbrev.unique()
    initAssign = True
    for g in gameList:
        
        #assign initial value at the beginning of the season
        if initAssign: #do only once
            for team in teamList:
                teamGames = dfs[dfs.teamAbbrev == team]      
                fg = teamGames.iloc[0]
                dfs.preGameELO[(dfs.gameId == fg.gameId) &\
                    (dfs.teamAbbrev == team)] = float(ELO_MEAN)
            initAssign = False

        #get two indices        
        hix = dfs[(dfs.gameId == g) & (dfs.gameLocationCode == 'H')].index[0]
        rix = dfs[(dfs.gameId == g) & (dfs.gameLocationCode == 'R')].index[0]
        
        #get team names and pre-ELO
        hTeam = dfs.ix[hix].teamAbbrev
        rTeam = dfs.ix[rix].teamAbbrev
        hELO = dfs.ix[hix].preGameELO
        rELO = dfs.ix[rix].preGameELO

        #expected score - home team
        dpower = (rELO - hELO - ELO_HOMEBONUS)/ELO_DELTA/2.
        expHome = 1.0/(1.0+np.power(10,dpower))
        dfs.gamePredict.ix[hix] = 1.0/(1.0+np.power(10,dpower))
        dfs.gamePredict.ix[rix] = 1.0/(1.0+np.power(10,-dpower))
        
        #calculate postGame
        homeScore = ts[(ts.gameId == g) &\
            (ts.teamAbbrev == dfs.ix[hix].teamAbbrev)]
        roadScore = ts[(ts.gameId == g) &\
            (ts.teamAbbrev == dfs.ix[rix].teamAbbrev)]
        
        #get goals for the home team
        homeWins = homeScore.wins.values[0]
        roadWins = roadScore.wins.values[0]
        homeOTLoss = homeScore.otLosses.values[0]
        roadOTLoss = roadScore.otLosses.values[0]

        #real score - home team
        if (homeWins == 1):
            realHome = 1.0
        elif (roadWins == 1):
            realHome = 0.0
        
        #check for OT:
        if (homeOTLoss == 1):
            realHome = 1.0-ELO_OTWIN_COEFF
        elif (roadOTLoss == 1):
            realHome = ELO_OTWIN_COEFF
        
        dfs.gameResult.ix[hix] = realHome
        dfs.gameResult.ix[rix] = 1.0 - realHome
        
        #post-game ELO rating
        deltaHome = ELO_K_FACTOR*(realHome-expHome)
        dfs.postGameELO.ix[hix] = hELO + deltaHome
        dfs.postGameELO.ix[rix] = rELO - deltaHome
        
        
        #assign forward to a new preGameELO
        homeFuture = dfs[(dfs.teamAbbrev == hTeam) & (dfs.gameId > g)]
        roadFuture = dfs[(dfs.teamAbbrev == rTeam) & (dfs.gameId > g)]
        
        if len(homeFuture) > 0:
            nhix = homeFuture.iloc[0].gameId
            dfs.preGameELO[(dfs.gameId == nhix) &\
                (dfs.teamAbbrev == hTeam)] = hELO + deltaHome
        
        if len(roadFuture) > 0:
            nrix = roadFuture.iloc[0].gameId
            dfs.preGameELO[(dfs.gameId == nrix) &\
                (dfs.teamAbbrev == rTeam)] = hELO - deltaHome

    return dfs
    
    
    
        

if __name__ == '__main__':
    
    #read dataset    
    ts = pd.read_sql('teams_stats_by_game',engine,index_col='index')
    gs = pd.read_sql('games_schedule',engine)
    
    #add seasonId for convenience
    gs['seasonId'] = gs.apply(lambda x: int(str(x.gameId)[0:4]),axis=1)
    
    #calculate ELO forward    
    dfs = process_elo_forward(2010)
    
    #draw residuals and correlation lines
    res1 = fitLine(dfs.preGameELO,dfs.postGameELO,alpha=0.05)
    res2 = fitLine(dfs.gamePredict,dfs.gameResult,alpha=0.05)