from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

import pandas as pd
from sqlalchemy import create_engine
pd.options.mode.chained_assignment = None

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhlstats')
conn = engine.connect()


if __name__ == '__main__':
    training = pd.read_sql('training',engine)
    RNNtrain = training[['gameId','gameDate','homeTeam','roadTeam','result']]