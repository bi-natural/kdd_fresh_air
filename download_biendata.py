"""
fetch Live Data (1 week ago) from biendata.com

1) Air Quality Data
https://biendata.com/competition/airquality/{city}/{start_time}/{end_time}/2k0d1d8
city = 'bj' (for Beijing) or 'ld' (for London)
time = 'YEAR-MONTH-DATE'
Example: https://biendata.com/competition/airquality/ld/2018-03-31-0/2018-04-01-17/2k0d1d8

2) Grid Meteorology Data
https://biendata.com/competition/meteorology/{city}_grid/{start_time}/{end_time}/2k0d1d8
city = 'bj' (for Beijing) or 'ld' (for London)
time = 'YEAR-MONTH-DATE'
Example：https://biendata.com/competition/meteorology/bj_grid/2018-03-29-0/2018-04-01-17/2k0d1d8
"""

import numpy as np
import pandas as pd
import download_url as http

import datetime as dt
import os
import grids_near_station as gr
import merge_grid_weather as mg

debug = True


def build_url(kind, city, date, hour, n_days=7):
    if not kind in ['meteorology', 'airquality']:
        ValueError("Invalid data name. should be (meteorology, airquality)")
        return

    if not city in ['ld', 'bj']:
        ValueError("Invalid city name. should be (bj, ld)")
        return

    date1 = dt.datetime.strptime('%s-%d'%(date, hour), '%Y-%m-%d-%H')
    date2 = date1 - dt.timedelta(days=n_days)
    date3 = dt.datetime.utcnow()
    if date2 > date3:
        date2 = date3

    start_date = date3.strftime('%Y-%m-%d-%H')
    end_date = date1.strftime('%Y-%m-%d-%H')

    if debug:
        print('start_date = {}'.format(start_date))
        print('end_date   = {}'.format(end_date))

    if kind == 'meteorology':
        url = 'https://biendata.com/competition/{}/{}_grid/{}/{}/2k0d1d8'.format(kind, city, start_date, end_date)
    else:
        url = 'https://biendata.com/competition/{}/{}/{}/{}/2k0d1d8'.format(kind, city, start_date, end_date)

    if debug:
        print(url)

    return url


def build_url_since(kind, city, since_date):
    if not kind in ['meteorology', 'airquality']:
        ValueError("Invalid data name. should be (meteorology, airquality)")
        return

    if not city in ['ld', 'bj']:
        ValueError("Invalid city name. should be (bj, ld)")
        return

    date2 = dt.datetime.utcnow()
    start_date = since_date.strftime('%Y-%m-%d-%H')
    end_date = date2.strftime('%Y-%m-%d-%H')

    if debug:
        print('start_date = {}'.format(start_date))
        print('end_date   = {}'.format(end_date))

    if kind == 'meteorology':
        url = 'https://biendata.com/competition/{}/{}_grid/{}/{}/2k0d1d8'.format(kind, city, start_date, end_date)
    else:
        url = 'https://biendata.com/competition/{}/{}/{}/{}/2k0d1d8'.format(kind, city, start_date, end_date)

    if debug:
        print('url = ', url)

    return url


def fetch(date, hour):

    for city in ['bj', 'ld']:
        for kind in ['meteorology', 'airquality']:
            url = build_url(kind, city, date, hour)

            response = http.get(url)

            with open('download/{}-{:d}_{}_{}.csv'.format(date, hour, city, kind), 'w') as f:
                if debug:
                    print(response.text)
                f.write(response.text)


def fetch_yesterday():

    today = dt.datetime.utcnow()
    yesterday = today - dt.timedelta(days=1)
    yesterday_date = yesterday.strftime('%Y-%m-%d')

    for city in ['bj', 'ld']:
        for kind in ['meteorology', 'airquality']:
            url = build_url(kind, city, yesterday_date, 23, 1)
            response = http.get(url)
            filename = 'download/{}-{:d}_{}_{}.csv'.format(yesterday_date, 0, city, kind)

            with open(filename, 'w') as f:
                if debug:
                    print('file: {}'.format(filename))
                f.write(response.text)


def get_yesterday_weather_aqi(city, include_weather=False, overwrite=False):

    today = dt.datetime.utcnow()
    since_date = today - dt.timedelta(days=2)
    since_date_str = since_date.strftime('%Y-%m-%d-%H')

    met_filename = 'download/{}_{}_{}.csv'.format(since_date_str, city, 'meteorology')
    if overwrite or not os.path.exists(met_filename):
        # download 'meteorology' data
        url = build_url_since('meteorology', city, since_date)
        response = http.get(url)

        if response.text == 'None':
            ValueError('Invalid response')
            return

        with open(met_filename, 'w') as f:
            if debug:
                print('file: {} -> {} bytes'.format(met_filename, len(response.text)))
            f.write(response.text)

    aqi_filename = 'download/{}_{}_{}.csv'.format(since_date_str, city, 'airquality')
    if overwrite or not os.path.exists(aqi_filename):
        # download 'airquality' data
        url = build_url_since('airquality', city, since_date)
        response = http.get(url)

        if response.text == 'None':
            ValueError('Invalid response')
            return

        with open(aqi_filename, 'w') as f:
            if debug:
                print('file: {} -> {} bytes'.format(aqi_filename, len(response.text)))
            f.write(response.text)

    # check if download is success
    if not os.path.exists(met_filename):
        FileNotFoundError("file not found. {}".format(met_filename))
        return

    if not os.path.exists(aqi_filename):
        FileNotFoundError("file not found. {}".format(aqi_filename))
        return

    # id,station_id,time,weather,temperature,pressure,humidity,wind_direction,wind_speed
    metro_df = pd.read_csv(met_filename, parse_dates=['time'], header=0,
                           names=['id', 'stationid', 'time', 'weather', 'temperature',
                                  'pressure', 'humidity', 'winddirection', 'windspeed'])

    # drop first 'id' column (useless)
    metro_df.drop(metro_df.columns[[0]], axis=1, inplace=True)

    if not include_weather:
        metro_df.drop(metro_df.columns[[2]], axis=1, inplace=True)

    # generate new column 'weekday' from datetime
    metro_df['weekday'] = metro_df['time'].dt.dayofweek

    # reorder
    if include_weather:
        reordered_list = ['stationid', 'time', 'weekday', 'temperature', 'pressure', 'humidity',
                          'winddirection', 'windspeed', 'weather']
    else:
        reordered_list = ['stationid', 'time', 'weekday', 'temperature', 'pressure', 'humidity',
                          'winddirection', 'windspeed']

    metro_df = metro_df[reordered_list]

    # mark all NA values with previous record value (ffill)
    metro_df.replace(999017.0, np.nan, inplace=True)
    metro_df.replace(999999.0, np.nan, inplace=True)
    metro_df.fillna(method='ffill', inplace=True)

    # id,station_id,time,PM25_Concentration,PM10_Concentration,NO2_Concentration,CO_Concentration,O3_Concentration,SO2_Concentration
    aqi_df = pd.read_csv(aqi_filename, parse_dates=['time'], header=0,
                         names=['id', 'stationid', 'time', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2'])

    # drop first 'id' column (useless)
    aqi_df.drop(aqi_df.columns[[0]], axis=1, inplace=True)

    # mark all NA values with previous record value (ffill)
    aqi_df.replace(999017.0, np.nan, inplace=True)
    aqi_df.replace(999999.0, np.nan, inplace=True)
    aqi_df.fillna(method='ffill', inplace=True)

    # convert to AirQuality stations
    # load grid, aqi (station) info
    grid_station, aqi_station = gr.get_station_data(city)

    df1 = pd.DataFrame()
    for idx, row in aqi_station.iterrows():
        df2 = mg.groupby_near_grids(idx, row, metro_df, include_weather=False)
        if df2 is not None:
            if debug and len(df1) == 0:
                print('Idx = {}, Records = {}'.format(idx, len(df2)))
            df1 = df1.append(df2, ignore_index=True)

    # merge
    full_df = df1.merge(aqi_df, how='left', on=['stationid', 'time'])
    full_df.fillna(method='ffill', inplace=True)

    if debug:
        print(full_df.head(5))

    return full_df

if __name__ == '__main__':
    #date = dt.datetime.strptime('2018-05-05-1', '%Y-%m-%d-%H')
    #week_ago = date - dt.timedelta(days=7)
    #print("date = {}, week_ago = {}".format(date, week_ago))

    #url = build_url('meteorology', 'ld', '2015-05-05', 5)

    #fetch('2018-05-08', 0)
    yesterday = get_yesterday_weather_aqi('bj', include_weather=False, overwrite=True)
