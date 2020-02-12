"""
Grid id => longitude, latitude
Station id => get nearest 9 Grids
"""

import pandas as pd
import numpy as np
import os

debug = False


def get_beijing_grid(filename='data/Beijing_grid_weather_station.csv'):
    if not os.path.exists(filename):
        ValueError("file Not found {}".format(filename))
        return

    df = pd.read_csv(filename, index_col=[0], header=None, names=['grid', 'latitude', 'longitude'])
    return df


def get_london_grid(filename='data/London_grid_weather_station.csv'):
    if not os.path.exists(filename):
        ValueError("file Not found {}".format(filename))
        return

    df = pd.read_csv(filename, index_col=[0], header=None, names=['grid', 'latitude', 'longitude'])
    return df


def get_beijing_aq(filename='data/Beijing_AirQuality_Stations.csv'):
    if not os.path.exists(filename):
        ValueError("file Not found {}".format(filename))
        return

    df = pd.read_csv(filename, index_col=[0], header=0, names=['stationid', 'longitude', 'latitude'])
    cols = df.columns.tolist()
    df = df[[cols[1]] + [cols[0]]]
    return df


def get_london_aq(filename='data/London_AirQuality_Stations.csv', predict_only=True):
    if not os.path.exists(filename):
        ValueError("file Not found {}".format(filename))
        return

    df = pd.read_csv(filename, index_col=[0], header=0, names=['stationid', 'api_data', 'need_predict', 'hist',
                                                               'latitude', 'longitude', 'sitetype', 'sitename'])

    df.fillna(False, inplace=True)
    if predict_only:
        df = df[df['need_predict'] == True]

    cols = df.columns.tolist()
    df = df[[cols[3]] + [cols[4]]]
    return df


def get_beijing_data():
    df1 = get_beijing_grid()
    df2 = get_beijing_aq()
    return df1, df2


def get_london_data():
    df1 = get_london_grid()
    df2 = get_london_aq()
    return df1, df2


def groupby_near_grids(idx, row, df, include_weather=True):
    df2 = df[df['stationid'].isin(row['grids'])]

    if debug:
        print('group_by: {} with {} grids {} data records'.format(idx, len(row['grids']), len(df2)))

    if len(df2) == 0:
        ValueError('Value not found. 0 results of *group-by*ing')
        return

    # drop grid's stationid column
    df2.drop(df2.columns[[0]], axis=1, inplace=True)

    if include_weather:
        def fn_max_occur(x):
            return x.value_counts().index[0]

        df2 = df2.groupby(['time']).agg({'temperature': np.average, 'pressure': np.average,
                                         'humidity': np.average, 'winddirection': np.average,
                                         'windspeed': np.average, 'weather': fn_max_occur})
    else:
        df2 = df2.groupby(['time']).agg({'temperature': np.average, 'pressure': np.average,
                                         'humidity': np.average, 'winddirection': np.average,
                                         'windspeed': np.average})

    # last column => AirQuality's stationid (added)
    df2 = df2.reset_index()
    df2['stationid'] = idx

    if debug:
        print('before reorder: ', df2.columns.tolist())
    cols = df2.columns.tolist()
    df2 = df2[[cols[-1]] + cols[0:-1]]

    if debug:
        print('after reorder: ', df2.columns.tolist())
    return df2


def merge_groupby_grids(idx, row, grid_df, aqi_df, include_weather=True):
    df2 = df[grid_df['stationid'].isin(row['grids'])]

    if debug:
        print('group_by: {} with {} grids {} data records'.format(idx, len(row['grids']), len(df2)))

    if len(df2) == 0:
        ValueError('Value not found. 0 results of *group-by*ing')
        return

    # drop grid's stationid column
    df2.drop(df2.columns[[0]], axis=1, inplace=True)

    if include_weather:
        def fn_max_occur(x):
            return x.value_counts().index[0]

        df2 = df2.groupby(['time']).agg({'temperature': np.average, 'pressure': np.average,
                                         'humidity': np.average, 'winddirection': np.average,
                                         'windspeed': np.average, 'weather': fn_max_occur})
    else:
        df2 = df2.groupby(['time']).agg({'temperature': np.average, 'pressure': np.average,
                                         'humidity': np.average, 'winddirection': np.average,
                                         'windspeed': np.average})

    # last column => AirQuality's stationid (added)
    df2 = df2.reset_index()
    df2['stationid'] = idx

    if debug:
        print('before reorder: ', df2.columns.tolist())
    cols = df2.columns.tolist()
    df2 = df2[[cols[-1]] + cols[0:-1]]

    if debug:
        print('after reorder: ', df2.columns.tolist())
    return df2


if __name__ == '__main__':
    if not os.path.exists('data/Beijing_grid_weather_station.csv'):
        ValueError("file not found {}".format('data/Beijing_grid_weather_station.csv'))
    else:
        df = get_beijing_grid('data/Beijing_grid_weather_station.csv')
        print(len(df), df.head(5))

    if not os.path.exists('data/London_grid_weather_station.csv'):
        ValueError("file not found {}".format('data/London_grid_weather_station.csv'))
    else:
        df = get_london_grid('data/London_grid_weather_station.csv')
        print(len(df), df.head(5))

    if not os.path.exists('data/Beijing_AirQuality_Stations.csv'):
        ValueError("file not found {}".format('data/Beijing_AirQuality_Stations.csv'))
    else:
        df = get_beijing_aq('data/Beijing_AirQuality_Stations.csv')
        print(len(df), df.head(5))

    if not os.path.exists('data/London_AirQuality_Stations.csv'):
        ValueError("file not found {}".format('data/London_AirQuality_Stations.csv'))
    else:
        df = get_london_aq('data/London_AirQuality_Stations.csv')
        print(len(df), df.head(5))
