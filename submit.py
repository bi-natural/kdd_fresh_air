"""
Submit KDD
"""

import datetime as dt
import pandas as pd
import requests

debug = True


def save_station(city, station, df):
    if df is None:
        return

    if debug:
        print('save on submit/station/{}_{}...'.format(city, station))

    today = dt.datetime.utcnow()
    today_str = today.strftime('%Y-%m-%d')

    df.to_csv('submit/station/{}_{}_{}.csv'.format(today_str, city, station), index=False)


def save_city(city, df):
    if df is None:
        return

    if debug:
        print('save on submit/city/DATE_{}_...'.format(city))

    today = dt.datetime.utcnow()
    today_str = today.strftime('%Y-%m-%d')

    df.to_csv('submit/city/{}_{}.csv'.format(today_str, city), index=False)


def save_all(df, count=0):
    if df is None:
        return

    if debug:
        print('save on submit/sent/DATE_...')

    today = dt.datetime.utcnow()
    today_str = today.strftime('%Y-%m-%d')

    if count == 0:
        target_file = 'submit/sent/{}.csv'.format(today_str)
    else:
        target_file = 'submit/sent/{}_{:d}.csv'.format(today_str, count)

    df.to_csv(target_file, index=False)

    return target_file


def sent(sent_file=None, count=0):

    if sent_file is None:
        today = dt.datetime.utcnow()
        today_str = today.strftime('%Y-%m-%d')
        if count == 0:
            sent_file = 'submit/sent/{}.csv'.format(today_str)
        else:
            sent_file = 'submit/sent/{}_{:d}.csv'.format(today_str, count)

        if debug:
            print('target file = {}'.format(sent_file))

    files = {
        'files': open(sent_file, 'rb')
    }

    data = {
        "user_id": "jaehoyang",
        "team_token": "4d46043f66974e11b397ab7524bcf279a0f79e879ac59ce31a96c2ac56bda922",
        "description": 'none',
        "filename": sent_file,
    }

    url = 'https://biendata.com/competition/kdd_2018_submit/'
    response = requests.post(url, files=files, data=data)
    print(response.text)


if __name__ == '__main__':
    sent(sent_file=None, count=2)
