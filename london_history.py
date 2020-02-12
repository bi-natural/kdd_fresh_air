"""
London History data (old)
"""

import datetime
import pandas as pd
import numpy as np
import os

import grids_near_station as gr
import merge_grid_weather as mg

debug = True


def build_history(grid_file='data/London_historical_meo_grid.csv',
                  aqi_file='data/London_historical_aqi_forecast_stations_20180331.csv',
                  out_file='data/London_historical_data.csv'):
    if not os.path.exists(grid_file):
        FileNotFoundError("file not found. {}".format(grid_file))
        return

    if not os.path.exists(aqi_file):
        FileNotFoundError("file not found. {}".format(aqi_file))
        return

    # id, station_id, forecast_time, weather, temperature, pressure, humidity, wind_speed, wind_direction
    grid_df = pd.read_csv(grid_file, parse_dates=['time'], header=0,
                          names = ['stationid', 'longitude', 'latitude', 'time', 'temperature',
                                   'pressure', 'humidity', 'winddirection', 'windspeed'])

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

    # null,MeasurementDateGMT,station_id,PM2.5 (ug/m3),PM10 (ug/m3),NO2 (ug/m3)
    aqi_df = pd.read_csv(aqi_file, parse_dates=['time'], header=0,
                          names = ['id', 'time', 'stationid', 'PM25', 'PM10', 'NO2'])

    # drop 'id' (useless)
    aqi_df.drop(aqi_df.columns[[0]], axis=1, inplace=True)

    if debug:
        print(grid_df.head(5))
        print(aqi_df.head(5))

    # load grid, aqi (station) info
    grid_station, aqi_station = gr.get_station_data('ld')

    if debug:
        print('-------------- successfully get_station_data(ld) ----------- ')
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


def convert_data(in_file='data/LondonData_temp.csv',
                 out_file='data/KDD_ld_hist'):
    if not os.path.exists(in_file):
        ValueError("file Not found {}".format(in_file))
        return

    # read data
    df = pd.read_csv(in_file,  parse_dates=['DateTime'])

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

    df.rename(columns={'StationID': 'stationid',
                       'DateTime': 'time',
                       'windspeedkph': 'windspeed'}, inplace=True)

    # 'time' should be exist
    if 'time' not in df.columns.tolist():
        AttributeError('cannot find <time> attribute name')
        return

    # generate new column 'weekday' from datetime
    df['weekday'] = df['time'].dt.dayofweek

    # reorder
    #     (stationid=1) + (utctime=0) + (weekday=-1) + (weather...) + (pollutants)
    cols = df.columns.tolist()
    df = df[['stationid', 'time', 'weekday',
             'temperature', 'pressure', 'humidity', 'winddirection', 'windspeed',
             'PM25', 'PM10', 'NO2']]

    # mark all NA values with each mean()
    df.replace(999017.0, np.nan, inplace=True)
    df.replace(999999.0, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # sort by 'stationid' + 'time'
    df.sort_values(by=['stationid', 'time'])

    # summarize first 5 rows
    if debug:
        print(df.head(5))

    # save to file
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    build_history()