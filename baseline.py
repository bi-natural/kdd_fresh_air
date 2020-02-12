"""
baseline prediction for Time Series forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

debug = True


def load_historical_data(file='data/Beijing_historical_data.csv', include_weather=False):
    """ Attribute Information: (input)
                            original     w/o stationid              --> reorder for predict
    :stationid: station id (0) string
    :weekofday:                          (0) inserted               (0) weekofday
    :time: date format     (1) datetime  (1) HH only                (1) HH
    :temperature: real     (2) real      (2)                        (2) PM25     --> Predict
    :pressure: real        (3) real      (3)                        (3) PM10     --> Predict
    :humidity: real        (4) real      (4)                        (4) O3       --> Predict
    :winddirection: real   (5) real      (5)                        (5) NO2      --> Predict
    :windspeedkph: real    (6) real      (6)                        (6) CO       --> Predict
    :PM25: real            (7) real      (7) => predict  =(-6)      (7) SO2      --> Predict
    :PM10: real            (8) real      (8) => predict  =(-5)      (8) temperature
    :NO2: real             (9) real      (9) => predict  =(-4)      (9) pressure
    :CO: real             (10) real     (10) => predict  =(-3)      (10) humidity
    :O3: real             (11) real     (11) => predict  =(-2)      (11) winddirection
    :SO2: real            (12) real     (12) => predict  =(-1)      (12) windspeed
    """

    if not os.path.exists(file):
        ValueError("file Not found {}".format(file))
        return

    # load dataset
    if include_weather:
        column_list = ['stationid', 'time', 'temperature', 'pressure', 'humidity',
                       'winddirection', 'windspeed', 'weather', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    else:
        column_list = ['stationid', 'time', 'temperature', 'pressure', 'humidity',
                       'winddirection', 'windspeed', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2']

    df = pd.read_csv(file, parse_dates=['time'], header=0, names=column_list)

    # generate new column 'weekday' from datetime
    df['weekday'] = df['time'].dt.dayofweek

    # reorder
    #     (stationid=1) + (utctime=0) + (weekday=-1) + ...
    if include_weather:
        reordered_list = ['stationid', 'time', 'weekday', 'PM25', 'PM10', 'O3', 'NO2', 'CO', 'SO2',
                          'temperature', 'pressure', 'humidity', 'winddirection', 'windspeed', 'weather']
    else:
        reordered_list = ['stationid', 'time', 'weekday', 'PM25', 'PM10', 'O3', 'NO2', 'CO', 'SO2',
                          'temperature', 'pressure', 'humidity', 'winddirection', 'windspeed']

    df = df[reordered_list]

    # mark all NA values with previous record value (ffill)
    df.replace(999017.0, np.nan, inplace=True)
    df.replace(999999.0, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # drop left NaN
    df.dropna(inplace=True)

    return df


def show_simple():
    df = load_historical_data(include_weather=False)

    df.plot(x='time', y=['PM25'])
    df.plot(x='time', y=['PM10'])
    plt.show()


def persistence_model():
    df = load_historical_data(include_weather=False)

    # test for PM25
    df = df[['PM25']]
    df = df.head(100)

    # create lagged dataset
    if debug:
        print(df.head(5))

    df2 = pd.concat([df.shift(1), df], axis=1)
    df2.columns = ['t-1', 't']

    if debug:
        print(df2.head(5))

    # split train/test (ignore first NaN value)
    size = int(len(df2) * 0.667)
    df2 = df2.values
    train, test = df2[1:size, :], df2[size:, :]
    X_tr, y_tr = train[:, 0], train[:, 1]
    X_te, y_te = test[:, 0], test[:, 1]

    # persistent_model
    def model_persistence(x):
        return x

    # walk-forward validation
    pred = list()
    for x in X_te:
        yhat = model_persistence(x)
        pred.append(yhat)

    te_score = mean_squared_error(y_te, pred)
    print('Test MSE: {:.3f}'.format(te_score))

    # plot
    plt.plot(y_tr)
    plt.plot([None for i in y_tr] + [x for x in y_te])
    plt.plot([None for i in y_tr] + [x for x in pred])
    plt.show()


def stationary_test(div=10, column='PM25'):
    df = load_historical_data(include_weather=False)
    df = df[[column]]

    # split N parts
    size = int(len(df) / div)

    i = 0
    while i < len(df):
        j = min(i + size, len(df))
        df2 = df[i:j].values
        if len(df2) > 10:
            mean = df2.mean()
            var = df2.var()
            print('#{:06d} ~ #{:06d}: mean={:.3f}, variance={:.3f}'.format(i, j, mean, var))
        i = j


def augmented_dickey_fuller_test(column='PM25'):
    from statsmodels.tsa.stattools import adfuller

    df = load_historical_data(include_weather=False)
    df = df[[column]].head(5000)

    result = adfuller(df.iloc[:, 0].values, autolag='AIC')

    print('ADF Statistic: {:.3f}'.format(result[0]))
    print('p-value: {:.6f}'.format(result[1]))
    print('Critical values:')
    for key, value in result[4].items():
        print('\t{}: {:.6f}'.format(key, value))


def ts_decompose(column='PM25'):
    from statsmodels.tsa.seasonal import seasonal_decompose

    df = load_historical_data(include_weather=False)
    df = df[[column]].head(24*100)

    if debug:
        print(df.head(100))

    result = seasonal_decompose(df.values, model='additive', freq=24)
    result.plot()
    plt.show()


def test_stationarity(column='PM25', diff=0):
    from statsmodels.tsa.stattools import adfuller

    df = load_historical_data(include_weather=False)
    df = df[[column]].head(24*14)

    if diff > 0:
        df['diff'] = df[column] - df[column].shift(diff)
        df = df[['diff']]
        df.dropna(inplace=True)

    series = df.iloc[:, 0].values

    roll_mean = df.rolling(24).mean()
    roll_std = df.rolling(24).std()

    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.plot(series, color='blue', label='Original')
    plt.plot(roll_mean, color='red', label='Rolling Mean')
    plt.plot(roll_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    result = adfuller(series, autolag='AIC')

    print('ADF Statistic: {:.3f}'.format(result[0]))
    print('p-value: {:.6f}'.format(result[1]))
    print('Critical values:')
    for key, value in result[4].items():
        print('\t{}: {:.6f}'.format(key, value))


def show_acf_pacf(column='PM25', freq=24, diff=0):
    import statsmodels.graphics.tsaplots as tsplot

    df = load_historical_data(include_weather=False)
    df = df[[column]].head(24*14)

    if diff > 0:
        df['diff'] = df[column] - df[column].shift(diff)
        df = df[['diff']]
        df.dropna(inplace=True)

    series = df.iloc[freq:, 0].values

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    tsplot.plot_acf(series, lags=40, ax=ax1, alpha=0.05)
    ax2 = fig.add_subplot(212)
    tsplot.plot_pacf(series, lags=40, ax=ax2, alpha=0.05)

    plt.show()


if __name__ == '__main__':
    #persistence_model()
    stationary_test(10, 'PM25')
    #augmented_dickey_fuller_test('PM25')
    #test_stationarity('PM25', diff=0)
    #test_stationarity('PM25', diff=1)
    show_acf_pacf('PM25', 24, 1)
    #ts_decompose('PM25')
    #ts_decompose('temperature')
