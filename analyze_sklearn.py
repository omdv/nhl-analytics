#TODO: time-series classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import ensemble
from sklearn import learning_curve
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine


pd.options.mode.chained_assignment = None  #suppress chained assignment warning

#postgres
engine = create_engine(
    'postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()

#random forest classifier
#input - X and y
#output
def RandomForest(X,y,n_est,factors):

    #-------------------------------------------    
    #Random forest classifier
    forest = ensemble.RandomForestClassifier(oob_score=True, 
                                             n_estimators=n_est)
    forest.fit(X,y)
    
    #-------------------------------------------    
    #feature importance
    feat_impo = forest.feature_importances_
    feat_impo = 100.0*(feat_impo / feat_impo.max())
    feat_list = X.columns.values

    fi_threshold = 30
    important_idx = np.where(feat_impo > fi_threshold)[0]
 
     #Create a list of all the feature names above the importance threshold
    important_features = feat_list[important_idx]
 
    #Get the sorted indexes of important features
    sorted_idx = np.argsort(feat_impo[important_idx])[::-1]
     
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feat_impo[important_idx][sorted_idx[::-1]], align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    plt.show()
    #-------------------------------------------    
    
    return forest

#source - sklearn
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, n_jobs=1):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    
    train_sizes=np.linspace(0.1, 1.0, 5)
    
#    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#        X, y, test_size=0.25, random_state=0)
        
#    cv = cross_validation.Shuffle2Split(X_train.shape[0], n_iter=100, 
#                                       test_size=.25, random_state=0)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim([0,1.2])
    
    train_sizes, train_scores, test_scores = learning_curve.learning_curve(
        estimator, X, y, cv=5, 
        n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc=1)
    
    return plt
    
#source - sklearn
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, factors, title='Confusion matrix', 
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, factors, rotation=45)
    plt.yticks(tick_marks, factors)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    

if __name__ == '__main__':
    
    #read dataset    
    full = pd.read_sql('training',engine)
#    full = full[full.gameId > 2006020000].dropna()
    full = full.dropna()
    full = full[full.so == 0]
    
#    full['homeTeamName'] = full.homeTeam.factorize()[0]
#    full['roadTeamName'] = full.roadTeam.factorize()[0]
    
    #features to try
    selectFeatures = ['homeGoalieSavePctgAllMean',
                      'roadGoalieSavePctgAllMean',
                      'homeCntShotsLastMean',
                      'roadCntShotsLastMean',
                      'homeWngShotsLastMean',
                      'roadWngShotsLastMean',
                      'homeDefShotsLastMean',
                      'roadDefShotsLastMean']
#    create X and y
#    X = full.loc[:,'home':]
    X = full[selectFeatures]
#    X = full.ix[:,'homeGoalieShotsAgainstAllMean':'roadCntShotsLastMean']
        
    [full['resFct'],factors] = full.result.factorize()
    y = full.resFct
    
    for f in selectFeatures:
        X[f] = (X[f]-X[f].mean())/(X[f].max()-X[f].min())
    X['home'] = X['homeGoalieSavePctgAllMean']+X['homeCntShotsLastMean']+X['homeWngShotsLastMean']+X['homeDefShotsLastMean']
    X['road'] = X['roadGoalieSavePctgAllMean']+X['roadCntShotsLastMean']+X['roadWngShotsLastMean']+X['roadDefShotsLastMean']
    
    
    #split dataset
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=42)
    
    #plot 2D dataset
    from matplotlib.colors import ListedColormap
    h=0.02
    
    x_min, x_max = X['home'].min()-.1, X['home'].max()+.1
    y_min, y_max = X['road'].min()-.1, X['road'].max()+.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot()
    # Plot the training points
    ax.scatter(X_train['home'], X_train['road'], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test['home'], X_test['road'], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
##    #Random forest
#    n_estimators = 50
#    title = 'Random Forest: '+str(n_estimators)+ \
#    ' est. and '+str(X.shape[1])+' feat.'
#    
##    classifier = RandomForest(X_train, y_train, n_estimators, factors)
#    classifier = SVC(kernel="linear", C=0.025)
#    classifier.fit(X_train,y_train)
#    
#    #---------------------------------------    
#    #Learning curve    
#    plot_learning_curve(classifier, title, X_train, y_train, n_jobs=1)
#    
#    #---------------------------------------
#    #Prediction
#    y_pred = classifier.predict(X_test)    
#    
#    #Evaluation
#    report = metrics.classification_report(y_test,y_pred,
#        target_names = factors)
#        
#    #-------------------------------------------    
#    #Compute confusion matrix
#    cm = metrics.confusion_matrix(y_test, y_pred)
#    title = 'Confusion matrix: '+str(n_estimators)+ \
#    ' est. and '+str(X.shape[1])+' feat.'
#    plt.figure()
#    plot_confusion_matrix(cm,factors,title)
#    #-------------------------------------------