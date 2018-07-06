#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import date

# 读取数据
off_test = pd.read_csv('data/ccf_offline_stage1_test_revised.csv', header=None, keep_default_na=False)
off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
off_train = pd.read_csv('data/ccf_offline_stage1_train.csv', header=None)
off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
on_train = pd.read_csv('data/ccf_online_stage1_train.csv', header=None)
on_train.columns = ['user_id', 'merchant_id', 'action', 'coupon_id', 'discount_rate', 'date_received', 'date']

# 数据滑窗处理
dataset3 = off_test
feature3 = off_train[((off_train.date >= '20160315') & (off_train.date < '20160630')) | (
        (off_train.date == 'null') & (off_train.date_received >= '20160315') & (off_train.date_received < '20160630'))]

dataset2 = off_train[((off_train.date >= '20160515') & (off_train.date < '20160615')) | (
        (off_train.date == 'null') & (off_train.date_received >= '20160515') & (off_train.date_received < '20160615'))]
feature2 = off_train[((off_train.date >= '20160201') & (off_train.date < '20160514')) | (
        (off_train.date == 'null') & (off_train.date_received >= '20160201') & (off_train.date_received < '20160514'))]

dataset1 = off_train[((off_train.date >= '20160414') & (off_train.date < '20160514')) | (
        (off_train.date == 'null') & (off_train.date_received >= '20160414') & (off_train.date_received < '20160514'))]
feature1 = off_train[((off_train.date >= '20160101') & (off_train.date < '20160413')) | (
        (off_train.date == 'null') & (off_train.date_received >= '20160101') & (off_train.date_received < '20160413'))]


def get_day_of_week(s):
    return date(int(s[0, 4]), int(s[4, 6]), int(s[6, 8])).weekday() + 1


# 提取优惠券特征
def coupon_feature(feature):
    feature['day_of_week'] = feature.date_received.astype('str').apply(get_day_of_week)
    feature['day_of_month'] = feature.date_received.astype('str').apply(lambda x: int(x[6, 8]))
