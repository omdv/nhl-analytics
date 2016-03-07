import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#classifiers
from sklearn import linear_model                                                                                                                                              
from sklearn import datasets                                                                                                                                                  
from sklearn import metrics
from sklearn import ensemble
from sklearn import learning_curve
from sklearn import cross_validation

from sqlalchemy import create_engine

pd.options.mode.chained_assignment = None  #suppress chained assignment warning

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()

#random forest classifier for the dataset
def RandomForest(df):    
    
    #shuffle dataset
    dftrim = df[df.gameId > 2012020000]
    dftrim['win'] = dftrim.apply(lambda row: 
        [2 if row.win ==0.5 else row.win],axis=1)
    dftrim = dftrim.apply(np.random.permutation)
    
    #splitting
    q1 = int(len(df)*0.6)
    q2 = int(len(df)*0.8)
    q3 = len(df)
    
    #split in sets
    train = df[:q1]
    val = df[q1:q2]
    test = df[q2:q3]
    
    #make X and y
    X_full = dftrim[['homeWins','roadWins']]
    y_full = dftrim.win
    X_train = train[['homeWins','roadWins']]
    y_train = train.win

    #Random forest classifier
    forest = ensemble.RandomForestClassifier(oob_score=True, n_estimators=50)
    forest.fit(X_train,y_train)
    y_val_pred = forest.predict(val[['homeWins','roadWins']])
    y_test_pred = forest.predict(test[['homeWins','roadWins']])
    
    #learning curve
    train_sizes, train_scores, test_scores = learning_curve.learning_curve(
    forest, X_full, y_full, cv=10, n_jobs=-1, 
    train_sizes=np.linspace(.1, 1., 10), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title('RandomForestClassifier')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.grid()
    
    # Plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", 
             label="Training")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", 
             label="Validation")
    
    # Plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, color="r")
    
    # Draw the plot and reset the y-axis
    plt.legend(loc=1)    
    plt.draw()
    plt.show()
    
    return [forest, val.win,y_val_pred, test.win,y_test_pred]



if __name__ == '__main__':
    
    #read dataset    
    full = pd.read_sql('training',engine)
    res = RandomForest(full)
        
#    print str(res[1]) +' and '+ str(res[2])
