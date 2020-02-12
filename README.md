# 2018 KDD Cup of Fresh Air

ACM(Association for Computing Machinery)에서 주최하는 KDD(Knowledge Discovery in Database)의 
경진대회인 KDD Cup에 대한 참가 후기

## KDD CUP 2018 : Fresh air

올해(2018년)의 경쟁주제는 도심 대기의 오염도(air pollution)을 예측하는 모델을 만들어 보는 것임. 
베이징과 런던의 48개 도심에서의 향후 48시간의 PM 2.5, PM 10, O3에 대한 농도를 예측함.
5월 한달간 매일마다 향후 48시간에 대한 것을 예측하여 2일후에 실제 값과 비교하여 정확도를 측정하고
이를 점수화하여 순위 확정

## Data

2017년 1월부터 2018년 1월까지 13개월의 기상 데이터 및 오염물질 측정 데이터를 학습데이터로 사용하도록 오픈함.
  * 기상데이터 종류: 온도, 습도, 압력, 바람 방향, 바람의 세기, 날씨 
  * 오염 물질 데이터: PM2.5, PM10, O3, NO2, CO, SO2
  
## 모델 및 학습

Time series 데이터이기 때문에 필요한 데이터 전처리를 수행하였고, Deep Learning 모델로는 LSTM 를 사용하였음.
간단하게 Keras의 LSTM를 사용하여 48시간 후의 오염 물질 데이터값을 추출함.

## Test

향후 48시간의 데이터 예측을 위해 기상데이터에 대해서는 날씨정보 사이트에서 정보를 받아서 오염물질 데이터를 
예측하였음.

## 참고

beijing과 london의 데이터 차이의 특징및 약간 달라 beijing 모델과 london 모델이 소스코드는 많은 부분
일치하지만 별도의 python 코드로 분리되어 있음. 코드 리팩토링 등이 필요하지만 그냥 올림. ^^# kdd_fresh_air
