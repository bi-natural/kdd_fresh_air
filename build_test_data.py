"""
Build test_data for forecast
"""

import datetime as dt
import download_biendata as bein
import download_weather as caiyun

debug = True


def build_test_data(city, n_steps=6, include_weather=False):

    if city not in ['bj', 'ld']:
        ValueError('Invalid city = {}. should be [bj, ld]'.format(city))
        return

    yesterday = bein.get_yesterday_weather_aqi(city, include_weather=False, overwrite=True)
    weather = caiyun.get_next_48hour_weather(city, include_weather=False)

    if yesterday is None:
        print('Error occur on downloading yesterday weather + air_quality')
        return
    elif debug:
        print('yesterday weather + air_quality data from beindata: {} records'.format(len(yesterday)))

    if weather is None:
        print('Error occur on downloading next 48hour weather forecast')
        return
    elif debug:
        print('next 48hour weather forecast data from caiyunapp: {} records'.format(len(weather)))

    if debug:
        print('-----------------build_test_data------------------')
        print(len(yesterday), yesterday.tail(10))
        print(len(weather), weather.head(10))

    weather_min_date = weather['time'].min()
    date1 = weather_min_date - dt.timedelta(hours=n_steps)

    if debug:
        print('clip_date -> {} ~ {}'.format(date1, weather_min_date))

    mask = (yesterday['time'] >= date1) & (yesterday['time'] < weather_min_date)
    yesterday = yesterday.loc[mask]

    if debug:
        print('-------------- yesterday ----------')
        sample_m = (yesterday['stationid'] == yesterday.iloc[0,:]['stationid'])
        sample = yesterday[sample_m]
        print(len(sample), sample)
        yesterday.to_csv('debug/ye.csv', index=False)

        print('-------------- weather ----------')
        sample_m = (yesterday['stationid'] == yesterday.iloc[0,:]['stationid'])
        sample = weather[sample_m]
        print(len(sample), sample)
        weather.to_csv('debug/we.csv', index=False)

    # merge
    test_data = yesterday.append(weather)
    test_data['weekday'] = test_data['time'].dt.dayofweek

    # reorder
    #     (stationid=1) + (utctime=0) + (weekday=-1) + ...
    if include_weather:
        reordered_list = ['stationid', 'time', 'weekday', 'PM25', 'PM10', 'O3', 'NO2', 'CO', 'SO2',
                          'temperature', 'pressure', 'humidity', 'winddirection', 'windspeed', 'weather']
    else:
        reordered_list = ['stationid', 'time', 'weekday', 'PM25', 'PM10', 'O3', 'NO2', 'CO', 'SO2',
                          'temperature', 'pressure', 'humidity', 'winddirection', 'windspeed']

    test_data = test_data[reordered_list]

    # use only HH (daytime hour)
    test_data.sort_values(by=['stationid', 'time'], inplace=True)
    test_data['time'] = test_data['time'].dt.hour

    # if city is London, drop last 3 columns (CO, O3, SO2)
    if city == 'ld':
        test_data.drop(['O3', 'CO',  'SO2'], axis=1, inplace=True)

    if debug:
        test_data.to_csv('debug/te.csv', index=False)

    return test_data


def test_bj():
    df = build_test_data('bj', 6)

    if debug:
        print('----------sample: {} test_data----------'.format('dongsi_aq'))
        sample_m = (df['stationid'] == 'dongsi_aq')
        sample = df[sample_m]
        print(len(sample), sample)


def test_ld():
    df = build_test_data('ld', 6)

    if debug:
        print('----------sample: {} test_data----------'.format('BL0'))
        sample_m = (df['stationid'] == 'BL0')
        sample = df[sample_m]
        print(len(sample), sample)


if __name__ == '__main__':
    test_ld()
