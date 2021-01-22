import pandas as pd
import numpy as np
from timeit import default_timer as timer
pd.options.plotting.backend = 'plotly'

train = pd.read_csv('train.csv')
states = train['Province_State'].unique()
state_dfs_raw = {state: train[train['Province_State'] == state] for state in states}
state_means = {}
state_stds = {}
state_dfs = {}
for s, state_df_raw in state_dfs_raw.items():
    state_df = state_df_raw[['Confirmed', 'Deaths', 'Date']]
    state_df['Date'] = pd.to_datetime(state_df['Date'], format='%m-%d-%Y')
    state_df = state_df.set_index('Date')

    mean, std = state_df.mean(), state_df.std()
    state_df = (state_df - mean) / std

    state_means[s] = mean
    state_stds[s] = std
    state_dfs[s] = state_df

column_names = state_dfs['Alabama'].columns