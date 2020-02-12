"""
Train KDD Cup 2018 Fresh Air
:Beijing: 35 stations
:London: 13 stations
"""

import datetime as dt
import pandas as pd
import beijing_train
import london_train
import submit


def train_all(send=True):
    bj = beijing_train.train_eval(100)
    ld = london_train.train_eval(100)

    bj = bj.append(ld, ignore_index=True)
    file = submit.save_all(bj)

    if send:
        submit.sent(sent_file=file)

def retry(count=1):
    today = dt.datetime.utcnow()
    today_str = today.strftime('%Y-%m-%d')
    sent_file = 'submit/sent/{}.csv'.format(today_str)

    x = pd.read_csv(sent_file)
    x.loc[x['PM2.5'] < 0, 'PM2.5'] = 0.01
    x.loc[x['PM10'] < 0, 'PM10'] = 0.01
    x.loc[x['O3'] < 0, 'O3'] = 0.01

    print(x)

    submit.save_all(x, count)

def sent_now():
    submit.sent(sent_file=None)


if __name__ == '__main__':
    retry(2)

