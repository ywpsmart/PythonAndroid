# -*- coding: utf-8 -*-

from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRectangleFlatButton, MDFlatButton
from kivy.lang import Builder
from helpers import code_helper
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.dialog import MDDialog

from kivy.core.window import Window

Window.clearcolor = (1, 1, 1, 1)
Window.size = (360, 600)
fontName = 'NanumGothicBold.ttf'

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import traceback
import datetime

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

krxUrl = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download'
stockUrl = 'http://finance.naver.com/item/sise_day.nhn?code={code}'
pageUrl = 'http://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'


class MainApp(MDApp):
    def build(self):
        screen = Screen()
        self.theme_cls.primary_palette = "Green"

        self.predict_start = False

        layout1 = BoxLayout(orientation='horizontal', spacing=10, padding=20)
        layout2 = BoxLayout(orientation='horizontal', spacing=10, padding=20)
        # layout3 = BoxLayout(orientation='horizontal', spacing=10, padding=20)
        layout4 = BoxLayout(orientation='vertical', padding=20, pos_hint={'center_x': .5, 'center_y': 0.7})

        self.code_input = Builder.load_string(code_helper)
        stockDataBtn = MDRectangleFlatButton(text='Data 수집', font_name=fontName,
                                             pos_hint={'center_x': .5, 'center_y': .9},
                                             size_hint_x=None, width=100, on_release=self.stockDataBtn_pressed)
        predictBtn = MDRectangleFlatButton(text='Predict', font_name=fontName,
                                           pos_hint={'center_x': .5, 'center_y': .9},
                                           size_hint_x=None, width=70, on_release=self.predictBtn_pressed)

        self.page_last = MDTextField(text='page_last', font_name=fontName,
                                     pos_hint={'center_x': .5, 'center_y': .8},
                                     size_hint_x=None, width=100)
        self.startDate = MDTextField(text='StartDate',
                                     pos_hint={'center_x': .5, 'center_y': .8},
                                     size_hint_x=None, width=100)
        self.endDate = MDTextField(text='EndDate',
                                   pos_hint={'center_x': .5, 'center_y': .8},
                                   size_hint_x=None, width=100)

        # splitBtn = MDRectangleFlatButton(text='Data 분류', font_name=fontName, pos_hint={'center_x': .5, 'center_y': .7},
        #                                  size_hint_x=None, width=70, on_release=self.splitBtn_pressed)
        #
        # trainBtn = MDRectangleFlatButton(text='Model훈련', font_name=fontName, pos_hint={'center_x': .5, 'center_y': .7},
        #                                  size_hint_x=None, width=70, on_release=self.trainBtn_pressed)

        # predictBtn = MDRectangleFlatButton(text='Predict', font_name=fontName,
        #                                    pos_hint={'center_x': .5, 'center_y': .7},
        #                                    size_hint_x=None, width=70, on_release=self.predictBtn_pressed)

        self.progress = MDTextField(text='진행상황', font_name=fontName,
                                    size_hint_x=None, width=300)

        self.high_cl = MDTextField(text='high_cl', font_name=fontName, font_size=14,
                                   size_hint_x=None, width=300)
        self.low_cl = MDTextField(text='low_cl', font_name=fontName, font_size=14,
                                  size_hint_x=None, width=300)
        self.max = MDTextField(text='max', font_name=fontName, font_size=14,
                               size_hint_x=None, width=300)
        self.min = MDTextField(text='min', font_name=fontName, font_size=14,
                               size_hint_x=None, width=300)
        self.avg = MDTextField(text='avg', font_name=fontName, font_size=14,
                               size_hint_x=None, width=300)

        layout1.add_widget(self.code_input)
        layout1.add_widget(stockDataBtn)
        layout1.add_widget(predictBtn)

        layout2.add_widget(self.page_last)
        layout2.add_widget(self.startDate)
        layout2.add_widget(self.endDate)

        # layout3.add_widget(splitBtn)
        # layout3.add_widget(trainBtn)
        # layout3.add_widget(predictBtn)

        layout4.add_widget(self.progress)
        layout4.add_widget(self.high_cl)
        layout4.add_widget(self.low_cl)
        layout4.add_widget(self.max)
        layout4.add_widget(self.min)
        layout4.add_widget(self.avg)

        screen.add_widget(layout1)
        screen.add_widget(layout2)
        # screen.add_widget(layout3)
        screen.add_widget(layout4)

        return screen

    def splitData(self, obj):
        code = self.code_input.text
        pg_last = int(self.page_last.text)
        str_datefrom = self.startDate.text

        df = crawlData(code, pg_last, str_datefrom)
        # reverse date
        df_reverse = df.iloc[::-1]
        df_reverse = df_reverse.reset_index(drop=True, inplace=False)

        # X, y 분리
        X = df_reverse[['종가', '전일비', '시가', '고가', '저가', '거래량']].copy()
        y = df_reverse[['종가']].copy()

        # X_train, X_test, y_train, y_test 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

        print(df.head())

        return X_train, X_test, y_train, y_test, df_reverse;

    def trainData(self, obj):
        self.progress.text = "Model training 시작"
        X_train, X_test, y_train, y_test, df_reverse = self.splitData(obj)

        # test에 사용할 past_60_days_before test data = X_train.tail(60)
        past_60_days_test = X_train.tail(60)

        # X_train, y_train --> Normalization
        scaler = MinMaxScaler()
        X_train_norm = scaler.fit_transform(X_train)
        y_train_norm = scaler.fit_transform(y_train)

        # y_train 61번째는 X_train 처음 60 days 를 기준으로 예측한 종가 이므로 60개씩 chunk.
        X_train_norm_chunk_np, y_train_norm_sequence_np = trainChunk(X_train_norm, y_train_norm)

        # model : build, compile, and fit
        regressor = model(X_train_norm_chunk_np)
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        h = regressor.fit(X_train_norm_chunk_np, y_train_norm_sequence_np, epochs=50, batch_size=32, verbose=1)

        # prepare test datasets
        X_test_added = past_60_days_test.append(X_test, ignore_index=True)

        # X_test_added, y_test --> Normalization
        X_test_added_norm = scaler.fit_transform(X_test_added)
        y_test_norm = scaler.fit_transform(y_test)

        X_test_norm_chunk_np, y_test_norm_sequence_np = testChunk(X_test_added_norm, y_test_norm)

        print(X_test_norm_chunk_np)
        self.progress.text = "Model training 완료"

        return scaler, y_test, df_reverse, regressor, X_test_norm_chunk_np, h;

    def predictBtn_pressed(self, obj):

        close_button = MDFlatButton(text='Close', on_release=self.close_dialog)
        # more_button = MDFlatButton(text='More')
        self.dialog = MDDialog(title='Predict', text='training start !!',
                               size_hint=(.7, 1), buttons=[close_button])
        self.dialog.open()

        # if self.predict_start:
        scaler, y_test, df_reverse, regressor, X_test_norm_chunk_np, h = self.trainData(obj)

        y_test_pred_norm_np = regressor.predict(X_test_norm_chunk_np)

        # back to original values : y_test_pred_norm_np, loss
        y_test_pred_back_np = scaler.inverse_transform(y_test_pred_norm_np)
        loss = [h.history['loss']]
        loss_np = np.array(loss)
        loss_value = scaler.inverse_transform(loss_np)
        loss_last = np.math.sqrt(loss_value[-1][-1])

        # label 인 y_test 의 nparray
        y_test_np = np.array(y_test)

        # Modification of prediction
        diff_back = y_test_np - y_test_pred_back_np
        # LR = diff 와 y_test_pred_np 의 선형함수
        LR = LinearRegression()
        LR.fit(y_test_pred_back_np, diff_back)
        diff_pred_back = LR.predict(y_test_pred_back_np)

        # 전일 종가
        price_close = y_test['종가'].values[y_test.shape[0] - 1]

        # 방식1 (y_pred_backBYdiff) : 실측치와 예측치의 차이를 예측치에 Add하여 보완
        y_pred_backBYdiff = y_test_pred_back_np + diff_pred_back
        # 방식2 (y_pred_backBYdiffMean) : 실측치와 예측치의 차이의 평균을 예측치에 Add하여 보완
        diff_back_mean = diff_pred_back.mean()
        y_pred_backBYdiffMean = y_test_pred_back_np + diff_back_mean
        # 방식 3 신뢰구간 95% _ 최대, 최소 예측치
        pred_C95_high = int(y_test_pred_back_np[-1][-1] + 2 * loss_last)
        pred_C95_low = int(y_test_pred_back_np[-1][-1] - 2 * loss_last)

        # Prediction of tomorrow _ diff & mean
        pred_y_tomorrow = int(y_test_pred_back_np[-1][-1])
        pred_y_tomorrowBYdiff = int(y_pred_backBYdiff[-1][-1])
        pred_y_tomorrowBYmean = int(y_pred_backBYdiffMean[-1][-1])
        pred_y_tomorrow_avg = int((pred_y_tomorrow + pred_y_tomorrowBYdiff + pred_y_tomorrowBYmean) / 3)

        pred_max = max(pred_y_tomorrow, pred_y_tomorrowBYdiff, pred_y_tomorrowBYmean, pred_y_tomorrow_avg)
        pred_min = min(pred_y_tomorrow, pred_y_tomorrowBYdiff, pred_y_tomorrowBYmean, pred_y_tomorrow_avg)
        pred_avg = int((pred_max + pred_min) / 2)

        # diff & mean 방식 - 구간 설정
        delta_max = pred_max - price_close
        delta_min = pred_min - price_close
        delta_avg = pred_avg - price_close

        # diff & mean 방식 - 등락률 설정
        str_p_max = '%0.2f' % ((delta_max / price_close) * 100)
        str_p_min = '%0.2f' % ((delta_min / price_close) * 100)
        str_p_avg = '%0.2f' % ((delta_avg / price_close) * 100)

        # 신뢰도 95% 방식 - 신뢰구간 설정
        delta_C95_high = pred_C95_high - price_close
        delta_C95_low = pred_C95_low - price_close

        # 신뢰도 95% 방식 - 등락률 설정
        str_p_C95_high = '%0.2f' % ((delta_C95_high / price_close) * 100)
        str_p_C95_low = '%0.2f' % ((delta_C95_low / price_close) * 100)

        # data_start & data_end
        date_first = df_reverse['날짜'].values[0]
        date_last = df_reverse['날짜'].values[df_reverse.shape[0] - 1]
        self.startDate.text = ""
        self.endDate.text = ""
        self.startDate.text = str(date_first)
        self.endDate.text = str(date_last)

        # # company name & 전일 종가
        # text.append('*** ' + self.lineEdit_name.text() + ' ***')
        # text.append(str(date_last) + ' : 종가 =' + str(price_close))
        #
        # # diff & mean 방식 결과
        # # text.append('max, min, avg : modified by diff regression')
        # # text.append('pred_max = ' + str(pred_max) + ',  등락 = ' + str(delta_max) + ' (' + str_p_max + '%)')
        # # text.append('pred_min = ' + str(pred_min) + ',  등락 = ' + str(delta_min) + ' (' + str_p_min + '%)')
        # # text.append('pred_avg = ' + str(pred_avg) + ',  등락 = ' + str(delta_avg) + ' (' + str_p_avg + '%)')

        # 신뢰도 95% 방식 결과
        # self.page_last.text = '신뢰구간 95% : model prediction +- 2 * std'

        self.progress.text = str(date_last) + ' : 종가 =' + str(price_close)

        self.high_cl.text = 'high_cl = ' + str(pred_C95_high) + ',  등락 = ' + str(
            delta_C95_high) + ' (' + str_p_C95_high + '%)'
        self.low_cl.text = 'low_cl = ' + str(pred_C95_low) + ',  등락 = ' + str(
            delta_C95_low) + ' (' + str_p_C95_low + '%)'
        self.max.text = 'max = ' + str(pred_max) + ',  등락 = ' + str(delta_max) + ' (' + str_p_max + '%)'
        self.min.text = 'min = ' + str(pred_min) + ',  등락 = ' + str(delta_min) + ' (' + str_p_min + '%)'
        self.avg.text = 'avg = ' + str(pred_avg) + ',  등락 = ' + str(delta_avg) + ' (' + str_p_avg + '%)'

    # def showBtn_pressed(self, obj):
    #     print(self.code_input.text)
    #     code = self.code_input.text
    #     self.stockData(code)
    #     return code

    def stockDataBtn_pressed(self, obj):
        code = self.code_input.text
        url = stockUrl.format(code=code)
        res = requests.get(url)
        res.encoding = 'utf-8'
        print('res.status_code =', res.status_code)
        if res.status_code == 200:
            soap = BeautifulSoup(res.text, 'lxml')
            el_table_navi = soap.find("table", class_="Nnavi")
            el_td_last = el_table_navi.find("td", class_="pgRR")
            pg_last = el_td_last.a.get('href').rsplit('&')[1]
            pg_last = pg_last.split('=')[1]
            pg_last = int(pg_last)
            print(pg_last)
            str_datefrom = datetime.datetime.strftime(datetime.datetime(year=2016, month=1, day=1), '%Y.%m.%d')
            str_dateto = datetime.datetime.strftime(datetime.datetime.today(), '%Y.%m.%d')

            self.page_last.text = str(pg_last)
            self.startDate.text = str_datefrom
            self.endDate.text = str_dateto

    def close_dialog(self, obj):
        self.progress.text = "training start !!"
        self.predict_start = True
        self.dialog.dismiss()


def crawlData(code, pg_last, str_datefrom):
    df = None
    for page in range(1, pg_last + 1):
        _df = parse_page(code, page)
        _df_filtered = _df[_df['날짜'] > str_datefrom]
        if df is None:
            df = _df_filtered
        else:
            df = pd.concat([df, _df_filtered])
        if len(_df) > len(_df_filtered):
            break
    return df


def model(X_train_norm_chunk_np):
    regressor = Sequential()

    regressor.add(LSTM(units=60, activation='relu', return_sequences=True,
                       input_shape=(X_train_norm_chunk_np.shape[1], X_train_norm_chunk_np.shape[2])))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=120, activation='relu'))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    return regressor


def testChunk(X_test_added_norm, y_test_norm):
    lookback = 60
    X_test_norm_chunk = []
    y_test_norm_sequence = []

    for i in range(lookback, X_test_added_norm.shape[0]):
        X_test_norm_chunk.append(X_test_added_norm[i - lookback:i])
        y_test_norm_sequence.append(y_test_norm[i - lookback, 0])

    X_test_norm_chunk_np = np.array(X_test_norm_chunk)
    y_test_norm_sequence_np = np.array(y_test_norm_sequence)

    return X_test_norm_chunk_np, y_test_norm_sequence_np


def trainChunk(X_train_norm, y_train_norm):
    lookback = 60
    X_train_norm_chunk = []
    y_train_norm_sequence = []

    for i in range(lookback, X_train_norm.shape[0]):
        X_train_norm_chunk.append(X_train_norm[i - lookback:i])
        y_train_norm_sequence.append(y_train_norm[i, 0])

    X_train_norm_chunk_np, y_train_norm_sequence_np = np.array(X_train_norm_chunk), np.array(y_train_norm_sequence)
    return X_train_norm_chunk_np, y_train_norm_sequence_np


def parse_page(code, page):
    try:
        url = pageUrl.format(code=code, page=page)
        res = requests.get(url)
        _soap = BeautifulSoup(res.text, 'lxml')
        _df = pd.read_html(str(_soap.find("table")), header=0)[0]
        _df = _df.dropna()
        return _df
    except Exception as e:
        traceback.print_exc()
    return None


if __name__ == '__main__':
    app = MainApp()
    app.run()
