"""
Fetch 48hour weather forecast data from Caiyun

Caiyun kindly provides 48 hours weather forecast data for all grid weather points.
http://kdd.caiyunapp.com/competition/forecast/{city}/{start_time}/2k0d1d8
city = ‘bj’ (for Beijing) or ‘ld’ (for London)
time = ‘YEAR-MONTH-DATE’
Example: http://kdd.caiyunapp.com/competition/forecast/bj/2018-04-24-00/2k0d1d8
"""

import numpy as np
import pandas as pd
import datetime as dt
import os
import download_url as http
import grids_near_station as gr

debug = True
desired_width = 250
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


def build_url(city, date):

    date1 = dt.datetime.strptime('%s'%(date), '%Y-%m-%d')
    date2 = date1 - dt.timedelta(days=1)
    yesterday = date2.strftime('%Y-%m-%d-23')

    url = 'https://kdd.caiyunapp.com/competition/forecast/{}/{}/2k0d1d8'.format(city, yesterday)
    if debug:
        print(url)
    return url


def fetch(city, date):
    url = build_url(city, date)

    response = http.get(url)

    if response.text != 'None':
        with open('fetch/{}_{}.csv'.format(city, date), 'w') as f:
            print(response.text)
            f.write(response.text)


def select_near_grids(idx, row, df, include_weather=True):

    #print(df.head(5))
    df2 = df[df['stationid'].isin(row['grids'])]
    #print("SELECT_NEAR_GRIDS = {}, {} -> {}".format(idx, row['grids'], len(df2)))
    #print(df2.head(5))

    # drop grid's stationid column
    df2.drop(df2.columns[[0]], axis=1, inplace=True)

    if include_weather:
        fn_max_occur = lambda x: x.value_counts().index[0]
        df2 = df2.groupby(['time']).agg({'temperature': np.average,
                                         'pressure': np.average,
                                         'humidity': np.average,
                                         'winddirection': np.average,
                                         'windspeed': np.average,
                                         'weather': fn_max_occur})
    else:
        df2 = df2.groupby(['time']).agg({'temperature': np.average,
                                         'pressure': np.average,
                                         'humidity': np.average,
                                         'winddirection': np.average,
                                         'windspeed': np.average})


    # last column => AirQuality's stationid (added)
    df2 = df2.reset_index()
    df2['stationid'] = idx
    #print(df2.columns.tolist())
    cols = df2.columns.tolist()
    df2 = df2[[cols[-1]] + cols[0:-1]]
    #print(df2.columns.tolist())
    return df2


def get_forcasted_weather(city, date, include_weather=True):
    if city not in ['bj', 'ld']:
        ValueError("Invalid city name. should be (bj, ld)")
        return

    raw_file = 'fetch/{}_{}.csv'.format(city, date)
    if not os.path.exists(raw_file):
        fetch(city, date)
        if not os.path.exists(raw_file):
            FileNotFoundError("file not found. maybe download failure")
            return

    # id, station_id, forecast_time, weather, temperature, pressure, humidity, wind_speed, wind_direction
    df = pd.read_csv(raw_file, parse_dates=['time'], header=0,
                     names=['id', 'stationid', 'time', 'weather', 'temperature',
                            'pressure', 'humidity', 'windspeed', 'winddirection'])

    # drop first 'id' column (useless)
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # drop if include_weather is False
    if not include_weather:
        df.drop(df.columns[[2]], axis=1, inplace=True)

    grid, aq = gr.get_station_data(city)

    df1 = pd.DataFrame()
    for idx, row in aq.iterrows():
        df2 = select_near_grids(idx, row, df, include_weather)
        if debug:
            print('CITY = {}, Idx = {}, Times = {}'.format(city, idx, len(df2)))
        df1 = df1.append(df2, ignore_index=True)

    print(df1.columns.tolist())
    print(len(df1), df1.head(5))

    out_file = 'test/{}_{}.csv'.format(city, date)
    df1.to_csv(out_file, index=False)

    return out_file


def get_next_48hour_weather(city, include_weather=False):

    today = dt.datetime.utcnow()
    today_date = today.strftime('%Y-%m-%d')

    out_filename = get_forcasted_weather(city, today_date, include_weather)

    # check if download is success
    if not os.path.exists(out_filename):
        FileNotFoundError("file not found. {}".format(out_filename))
        return

    # station_id,forecast_time,weather,temperature,pressure,humidity,wind_speed,wind_direction
    metro_df = pd.read_csv(out_filename, parse_dates=['time'], header=0)

    # mark all NA values with previous record value (ffill)
    metro_df.replace(999017.0, np.nan, inplace=True)
    metro_df.replace(999999.0, np.nan, inplace=True)
    metro_df.fillna(method='ffill', inplace=True)

    return metro_df


if __name__ == '__main__':
    get_forcasted_weather('ld', '2018-05-08', include_weather=False)
    get_forcasted_weather('bj', '2018-05-08', include_weather=False)
