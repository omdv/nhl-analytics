import json
import requests
import pandas as pd
import time
from sqlalchemy import create_engine

#postgres
engine = create_engine('postgresql://postgres:e2p71828@localhost:5432/nhlstats')
# conn = engine.connect()


if __name__ == '__main__':
    season = '20052006'
    games = pd.read_sql('games',engine)