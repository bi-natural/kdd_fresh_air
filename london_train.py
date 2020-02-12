"""
KDD Cup 2018 Fresh Air
:training: London historical data
"""

from math import sqrt
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, LSTM

import datetime as dt
import pandas as pd
import numpy as np
import os

import lstm_data as ld
import build_test_data as bt
import submit

debug = True
desired_width = 250
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


def show_dataframe(df, head=5):
    if head > 0:
        print('------------- Head({}) --------------'.format(head))
        print(df.head(head))

    print('------------- Columns -----------------')
    for col in df.columns.tolist():
        if df[col].dtype in ['float32', 'int32', 'float64', 'int64']:
            print('  {:15s} = {:10.3f} ~ {:10.3f}, NaN = {}'.format(col, df[col].min(), df[col].max(), df[col].isnull().sum()))


def load_historical_data(file='data/London_historical_data.csv', include_weather=False):
    """ Attribute Information: (input)
                            original     w/o stationid              --> reorder for predict
    :stationid: station id (0) string
    :weekofday:                          (0) inserted               (0) weekofday
    :time: date format     (1) datetime  (1) HH only                (1) HH
    :temperature: real     (2) real      (2)                        (2) PM25     --> Predict
    :pressure: real        (3) real      (3)                        (3) PM10     --> Predict
    :humidity: real        (4) real      (4)                        (4) NO2      --> Predict
    :winddirection: real   (5) real      (5)                        (5) temperature
    :windspeedkph: real    (6) real      (6)                        (6) pressure
    :PM25: real            (7) real      (7) => predict  =(-3)      (7) humidity
    :PM10: real            (8) real      (8) => predict  =(-2)      (8) winddirection
    :NO2: real             (9) real      (9) => predict  =(-1)      (9) windspeed
    """

    if not os.path.exists(file):
        ValueError("file Not found {}".format(file))
        return

    # load dataset
    if include_weather:
        column_list = ['stationid', 'time', 'temperature', 'pressure', 'humidity',
                       'winddirection', 'windspeed', 'weather', 'PM25', 'PM10', 'NO2']
    else:
        column_list = ['stationid', 'time', 'temperature', 'pressure', 'humidity',
                       'winddirection', 'windspeed', 'PM25', 'PM10', 'NO2']

    df = read_csv(file, parse_dates=['time'], header=0, names=column_list)

    # generate new column 'weekday' from datetime
    df['weekday'] = df['time'].dt.dayofweek

    # use only HH (daytime hour)
    df['time'] = df['time'].dt.hour

    # reorder
    #     (stationid=1) + (utctime=0) + (weekday=-1) + ...
    if include_weather:
        reordered_list = ['stationid', 'time', 'weekday', 'PM25', 'PM10', 'NO2',
                          'temperature', 'pressure', 'humidity', 'winddirection', 'windspeed', 'weather']
    else:
        reordered_list = ['stationid', 'time', 'weekday', 'PM25', 'PM10', 'NO2',
                          'temperature', 'pressure', 'humidity', 'winddirection', 'windspeed']

    df = df[reordered_list]

    # mark all NA values with previous record value (ffill)
    df.replace(999017.0, np.nan, inplace=True)
    df.replace(999999.0, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # drop left NaN
    df.dropna(inplace=True)

    # show dataFrame shortly
    # if debug:
    #    show_dataframe(df, 5)

    # n_features, n_ignore (last 5 attributes)
    n_features = len(df.columns.tolist()) - 1
    n_ignore = 5

    return df, n_features, n_ignore


def reframed_for_lstm(df, n_steps=6, n_features=10, n_ignore=5, include_weather=False):
    # get list of stations
    stations = df['stationid'].unique()
    if debug:
        print('reframed for LSTM')
        print('n_features = {}'.format(n_features))
        print('target {} stations: {}'.format(len(stations), stations))

    data = [ ]
    i = 0
    for station in stations:
        if debug and i == 0:
            print('#{}: station = {}'.format(i, station))

        df1 = df[df['stationid'] == station]
        df1 = df1.drop(columns=df1.columns[0], axis=1)

        if debug and i == 0:
            print('Shape = {}, Columns = {}'.format(df1.shape, df1.columns.tolist()))

        # integer encode weather (non integer/float type)
        if include_weather:
            encoder = LabelEncoder()
            df1.loc[:, 'weather'] = encoder.fit_transform(df1.loc[:, 'weather'])

        # ensure all data is float
        values = df1.values
        values = values.astype('float64')

        if debug and i == 0:
            dd = pd.DataFrame(values)
            print('before scaled = shape {}'.format(dd.shape))
            #print(dd.head(100))
            #print(dd.tail(200))
            #print('station: {} -> {}'.format(station, values[1, :]))

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(values)
        scaled = scaler.transform(values)

        # frame as supervised learning
        if debug and i == 0:
            dd = pd.DataFrame(scaled)
            print('scaled columns = #{} -> {}'.format(len(dd.columns.tolist()), dd.columns.tolist()))

        reframed = ld.series_to_supervised(scaled, n_steps, 1)
        if debug and i == 0:
            print("after reframed = shape {}".format(reframed.shape))
            #print(reframed.tail(5))

        # drop columns we don't want to predict (last utctime ~ weather)
        reframed.drop(reframed.columns[[range(-n_ignore, 0, 1)]], axis=1, inplace=True)
        if debug and i == 0:
            print('reframed columns = #{} -> {}'.format(len(reframed.columns.tolist()), reframed.columns.tolist()))
            print("after  re-framed.drop = shape {}".format(reframed.shape))
            #print(reframed.tail(5))

        item = {}
        item['station'] = station
        item['data'] = reframed
        item['scaler'] = scaler
        data.append(item)

        i += 1
    else:
        if debug:
            print('load {} stations data'.format(len(stations)))
        return data


class RNN_LSTM(object):
    def __init__(self, param_data, n_steps=6, n_features=10, n_ignore=5, n_nodes=50):
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_ignore = n_ignore
        self.param_data = param_data
        self.station_name = param_data['station']

        self.loading_data(param_data, n_steps, n_features, n_ignore)
        self.n_nodes = n_nodes
        self.model = self.build_model()

    def loading_data(self, item, n_steps=6, n_features=10, n_ignore=5):
        #
        # split into train and test sets
        #
        values = item['data'].values
        n_train_hours = values.shape[0] - 24*2
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]

        if debug:
            print('X: train = {}, test = {}'.format(train.shape, test.shape))

        # split into input and outputs
        n_obs = n_steps * n_features
        train_X, train_y = train[:, :n_obs], train[:, n_obs:]
        test_X, test_y = test[:, :n_obs], test[:, n_obs:]

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
        if debug:
            print('Fetch: X_tr = {}, y_tr = {}, X_te = {}, y_te = {}'.format(train_X.shape, train_y.shape,
                                                                             test_X.shape, test_y.shape))

        self.name = item['station']
        self.scaler = item['scaler']
        self.X_tr = train_X
        self.y_tr = train_y
        self.X_te = test_X
        self.y_te = test_y

    def build_model(self):
        model = Sequential()
        if debug:
            print('LSTM_Model: LSTM node = {}, input_shape = ({}, {})'.format(self.n_nodes,
                                                                              self.X_tr.shape[1],
                                                                              self.X_tr.shape[2]))

        model.add(LSTM(self.n_nodes, input_shape=(self.X_tr.shape[1], self.X_tr.shape[2])))
        #model.add(Dense(self.n_ignore))
        model.add(Dense(self.n_features-self.n_ignore))
        model.compile(loss='mae', optimizer='adam')
        return model

    def train(self, n_epochs=50, n_batch_size=72):
        history = self.model.fit(self.X_tr, self.y_tr, epochs=n_epochs, batch_size=n_batch_size,
                                 validation_data=(self.X_te, self.y_te),
                                 verbose=2, shuffle=False)
        return history

    def plot_train(self, history):
        pyplot.clf()
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.savefig('img/bj_{}_hist.png'.format(self.name))

    def plot_test(self, inv_y, inv_yhat, metrics):
        pyplot.clf()
        pyplot.figure(figsize=(9, 9))
        pyplot.suptitle('Station: %s, RMSE=%.3f, MAPE=%.3f, R2_Score=%.3f'%(self.name,
                                                                            metrics[0], metrics[1], metrics[2]))
        pollutants = ['PM2.5', 'PM10', 'NO2']
        i = 0
        for pollutant in pollutants:
            if i == 0:
                print('ax_plot_y : {}'.format(inv_y[:, i]))
                print('ax_plot_y^: {}'.format(inv_yhat[:, i]))
            ax = pyplot.subplot(3, 2, i+1)
            ax.plot(inv_y[:, i], label='y (label)')
            ax.plot(inv_yhat[:, i], label='^y (predict)')
            ax.set_title(pollutant)
            pyplot.legend()
            i += 1
        else:
            pyplot.savefig('img/bj_{}_test.png'.format(self.name))

        #print("inv_y")
        #print(inv_y.head(5))
        #print("inv_yhat")
        #print(inv_yhat.head(5))

    def validation(self, plot_image=True):
        y_hat = self.model.predict(self.X_te)
        if debug:
            dd = pd.DataFrame(y_hat)
            print('y_hat.shape= {}, value={}'.format(dd.shape, dd.values))

        self.X_te = self.X_te.reshape((self.X_te.shape[0], self.n_steps * self.n_features))

        # invert scaling for forecast
        inv_yhat = np.concatenate((y_hat, self.X_te[:, -self.n_ignore:]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, -self.n_features+2:-self.n_ignore]

        if debug:
            print('y_hat = {}, inv_yhat = {}'.format(y_hat.shape, inv_yhat.shape))

        # invert scaling for actual
        inv_y = np.concatenate((self.y_te, self.X_te[:, -self.n_ignore:]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -self.n_features+2:-self.n_ignore]

        if debug:
            print('y_te = {}, inv_y = {}'.format(self.y_te.shape, inv_y.shape))

        # calculate RMSE, MAPE, R2
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        mape = mean_absolute_error(inv_y, inv_yhat)
        r2   = r2_score(inv_y, inv_yhat)
        if debug:
            print('Test RMSE: %.3f, MAPE: %.3f, R2 Score: %.3f' % (rmse, mape, r2))

        # Show data
        if plot_image:
            self.plot_test(inv_y, inv_yhat, [rmse, mape, r2])

    def predict(self, forecast_data, hours=48):
        if debug:
            print('------------- predict 48hours --------------')
            print('Shape = {}, Columns = {}'.format(forecast_data.shape, forecast_data.columns.tolist()))
            print(forecast_data)

        # ensure all data is float
        forecast_data.fillna(0, inplace=True)
        forecast_values = forecast_data.values
        forecast_values = forecast_values.astype('float64')

        # normalize features
        scaled = self.scaler.transform(forecast_values)

        if debug:
            dd = pd.DataFrame(scaled)
            print('scaled rows= {}, columns= {} -> {}'.format(len(dd), len(dd.columns.tolist()), dd.columns.tolist()))

        for hr in range(0, hours):
            step_hr = ld.series_to_supervised(scaled[hr:hr+self.n_steps+1, :], self.n_steps, 1, dropnan=False)
            if debug and hr == 0:
                print('step_hr = {}, hour = #{} ~ #{}'.format(step_hr.shape, hr, hr+self.n_steps))

            # drop columns we don't want to predict (last utctime ~ weather)
            step_hr.drop(step_hr.columns[[range(-self.n_ignore, 0, 1)]], axis=1, inplace=True)

            # split into input (and outputs)
            n_obs = self.n_steps * self.n_features
            if debug and hr == 0:
                print('step_hr = {}, n_obs = {:d}'.format(step_hr.shape, n_obs))

            step_np = step_hr.values
            X_te = step_np[:, :n_obs]

            # reshape input to be 3D [samples, timesteps, features]
            X_te = X_te.reshape((X_te.shape[0], self.n_steps, self.n_features))

            # predict next 1 hour
            y_hat = self.model.predict(X_te)

            #if debug:
            #    print('#{:02d}: y_hat   = {}'.format(hr, y_hat[n_steps]))
            #    print('#{:02d}:+n_steps = {}'.format(hr, scaled[hr+n_steps, :]))
            scaled[hr+self.n_steps, 2:-self.n_ignore] = y_hat[self.n_steps, 2:]
            hr += 1
        else:
            # invert scaling for forecast
            scaled_back = self.scaler.inverse_transform(scaled)
            if debug:
                print('------- after predict --------')
                dd = pd.DataFrame(scaled_back)
                print(len(dd), dd)

            results = scaled_back[self.n_steps:, :-self.n_ignore]

            ft = pd.DataFrame(results, columns=['hour', 'weekday', 'PM2.5', 'PM10', 'NO2'])

            def gen_name_tag(row):
                return '{}#{}'.format(self.station_name, row.name)

            ft['test_id'] = ft.apply(gen_name_tag, axis=1)
            ft = ft[['test_id', 'PM2.5', 'PM10']]
            ft['O3'] = 0.

            # correct invalid output: output may not negative
            ft.loc[ft['PM2.5'] < 0, 'PM2.5'] = 0.01
            ft.loc[ft['PM10'] < 0, 'PM10'] = 0.01

            if debug:
                print('------- Prediction Data -------')
                print(len(ft), ft)

            return ft


def train_eval(n_epoch=100):
    if debug:
        print('######### Load Historical data for training ########')

    df, n_features, n_ignore = load_historical_data()

    if debug:
        print('##### n_features = {:d}, n_ignore = {:d}, records = {:d}'.format(n_features, n_ignore, len(df)))
        print(df.head(5))

    # split data by station
    n_steps = 6
    stations = reframed_for_lstm(df, n_steps, n_features, n_ignore)

    if debug:
        print('######### Build Forecasting Test data for submission ########')
    test_data = bt.build_test_data('ld', n_steps)

    if debug:
        print('###### forecast test: {}'.format(test_data.columns.tolist()))

    forecast_sum = pd.DataFrame()

    for station in stations:
        lstm = RNN_LSTM(station, n_steps, n_features, n_ignore, 200)
        history = lstm.train(n_epochs=n_epoch, n_batch_size=72)
        lstm.plot_train(history)
        lstm.validation()

        station_data = test_data[test_data['stationid'] == station['station']]
        station_data = station_data.drop(columns=station_data.columns[[0]], axis=1)

        if debug:
            print('###### Run Prediction: {}, Shape = {}'.format(station['station'], station_data.shape))

        forecast_result = lstm.predict(station_data, 48)
        submit.save_station('ld', station['station'], forecast_result)
        forecast_sum = forecast_sum.append(forecast_result, ignore_index=True)

    submit.save_city('ld', forecast_sum)
    return forecast_sum


if __name__ == '__main__':
    train_eval()
