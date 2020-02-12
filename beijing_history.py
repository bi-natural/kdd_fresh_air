"""
Beijing History data (old)
"""

import datetime as dt
import pandas as pd
import numpy as np
import os
import grids_near_station as gr
import merge_grid_weather as mg

debug = True
desired_width = 250
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


def get_min_max_date(margin, grid_min_date, grid_max_date, aqi_min_date, aqi_max_date):
    #
    min_date = grid_min_date if grid_min_date > aqi_min_date else aqi_min_date
    max_date = grid_max_date if grid_max_date < aqi_max_date else aqi_max_date

    date1 = min_date + dt.timedelta(hours=margin)
    date2 = max_date - dt.timedelta(hours=margin)
    return date1, date2


def build_history(grid_file='data/Beijing_historical_meo_grid.csv',
                  aqi_file='data/beijing_17_18_aq.csv',
                  out_file='data/Beijing_historical_data.csv'):
    if not os.path.exists(grid_file):
        FileNotFoundError("file not found. {}".format(grid_file))
        return

    if not os.path.exists(aqi_file):
        FileNotFoundError("file not found. {}".format(aqi_file))
        return

    # stationName,longitude,latitude,utc_time,temperature,pressure,humidity,wind_direction,wind_speed/kph
    grid_df = pd.read_csv(grid_file, parse_dates=['time'], header=0,
                          names = ['stationid', 'longitude', 'latitude', 'time', 'temperature',
                                   'pressure', 'humidity', 'winddirection', 'windspeed'])

    grid_min_date = grid_df['time'].min()
    grid_max_date = grid_df['time'].max()

    # drop (longitude, latitude) attribute if exists
    cols = grid_df.columns.tolist()
    length = len(cols)
    idx = []
    for i in range(0,length):
        if cols[i] in ['longitude', 'latitude']:
            idx.append(i)

    if len(idx) > 0:
        if debug:
            print("drop column: {}".format(idx))
        grid_df.drop(grid_df.columns[idx], axis=1, inplace=True)

    # stationId,utc_time,PM2.5,PM10,NO2,CO,O3,SO2
    aqi_df = pd.read_csv(aqi_file, parse_dates=['time'], header=0,
                         names = ['stationid', 'time', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2'])

    # drop 'id' (useless)
    # aqi_df.drop(aqi_df.columns[[0]], axis=1, inplace=True)

    aqi_min_date = aqi_df['time'].min()
    aqi_max_date = aqi_df['time'].max()

    # remove out-of-date
    min_date, max_date = get_min_max_date(48, grid_min_date, grid_max_date, aqi_min_date, aqi_max_date)

    if debug:
        print('Grid Date: {} ~ {}'.format(grid_min_date, grid_max_date))
        print('Aqi  Date: {} ~ {}'.format(aqi_min_date, aqi_max_date))
        print('*Commnon*: {} ~ {}'.format(min_date, max_date))

    grid_mask = (grid_df['time'] >= min_date) & (grid_df['time'] <= max_date)
    aqi_mask = (aqi_df['time'] >= min_date) & (aqi_df['time'] <= max_date)

    grid_df = grid_df.loc[grid_mask]
    aqi_df = aqi_df.loc[aqi_mask]

    if debug:
        print(grid_df.head(5))
        print(grid_df.tail(5))
        print(aqi_df.head(5))
        print(aqi_df.tail(100))

    # load grid, aqi (station) info
    grid_station, aqi_station = gr.get_station_data('bj')

    if debug:
        print('-------------- successfully get_station_data(bj) ----------- ')
        print(len(grid_station), grid_station.head(5))
        print(len(aqi_station), aqi_station.head(5))

    df1 = pd.DataFrame()
    for idx, row in aqi_station.iterrows():
        df2 = mg.groupby_near_grids(idx, row, grid_df, include_weather=False)
        if debug:
            print('Idx = {}, Times = {}'.format(idx, len(df2)))
        df1 = df1.append(df2, ignore_index=True)

    # merge with aqi_df
    if debug:
        print('===== df1 ====== ')
        print(df1.columns.tolist())
        print(len(df1), df1.head(5))
        print('===== aqi ====== ')
        print(aqi_df.columns.tolist())
        print(len(aqi_df), aqi_df.head(5))

    df1 = df1.merge(aqi_df, how='left', on=['stationid', 'time'])

    if debug:
        print('===== after merge ======')
        print(df1.columns.tolist())
        print(len(df1), aqi_df.head(5))

    df1.to_csv(out_file, index=False)


def convert_data(in_file='data/BeijingData_allstations.csv',
                 out_file='data/KDD_bj_hist'):
    if not os.path.exists(in_file):
        ValueError("file Not found {}".format(in_file))
        return

    # read data
    df = pd.read_csv(in_file,  parse_dates=['time'])

    cols = df.columns.tolist()
    length = len(cols)
    idx = []
    for i in range(0,length):
        if cols[i] in ['longitude', 'latitude']:
            idx.append(i)

    # drop (longitude, latitude) attribute if exists
    if len(idx) > 0:
        if debug:
            print("drop column: {}".format(idx))
        df.drop(df.columns[idx], axis=1, inplace=True)

    # only May(5) data
    #mask = (df['utctime'].dt.month == 5)
    #df = df.loc[mask]

    if 'time' not in df.columns.tolist():
        AttributeError('cannot find <time> attribute name')
        return

    # generate new column 'weekday' from datetime
    df['weekday'] = df['time'].dt.dayofweek

    # reorder
    #     (stationid=1) + (utctime=0) + (weekday=-1) + ...
    cols = df.columns.tolist()
    df = df[[cols[0]] + [cols[1]] + [cols[-1]] + cols[8:-1] + cols[2:8]]

    # mark all NA values with each mean()
    df.replace(999017.0, np.nan, inplace=True)
    df.replace(999999.0, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # summarize first 5 rows
    if debug:
        print(df.head(5))

    # save to file
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    build_history()
