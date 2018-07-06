#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        return -1  # those only receive once


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) - date(int(d[0:4]),
                                                                                                           int(d[4:6]),
                                                                                                           int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(int(date_received[0:4]), int(date_received[4:6]),
                                                                       int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def calc_discount_rate(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return float(s[0])
    else:
        return 1.0 - float(s[1]) / float(s[0])


def get_discount_man(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[0])


def get_discount_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[1])


def is_man_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 0
    else:
        return 1


def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                        int(s[1][6:8]))).days


def get_label(s):
    s = s.split(':')
    if s[0] == '0':
        return 0
    if s[0] == 'null':
        return 0
    elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                      int(s[1][6:8]))).days <= 15:
        return 1
    else:
        return -1


def covert_int(s):
    return int(s)


def other_feature(dataset):
    t = dataset[['user_id']]
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()  # 用户领取的所有优惠券数目

    t1 = dataset[['user_id', 'coupon_id']]
    t1['this_month_user_receive_same_coupon_count'] = 1  # 用户领取特定优惠券数目
    t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

    t2 = dataset[['user_id', 'coupon_id', 'date_received']]
    t2.date_received = t2.date_received.astype('str')
    t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset[['user_id', 'coupon_id', 'date_received']]
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype('int')
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype('int') - t3.min_date_received

    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
        is_firstlastone)
    t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
             'this_month_user_receive_same_coupon_firstone']]

    t4 = dataset[['user_id', 'date_received']]
    t4['this_day_user_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    t5 = dataset[['user_id', 'coupon_id', 'date_received']]
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()

    t6 = dataset[['user_id', 'coupon_id', 'date_received']]
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'}, inplace=True)

    t7 = dataset[['user_id', 'coupon_id', 'date_received']]
    t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)  #距离用户上一次领券的时间间隔
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)    #距离用户下一次领券的时间间隔
    t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

    other_feature = pd.merge(t1, t, on='user_id')
    other_feature = pd.merge(other_feature, t3, on=['user_id', 'coupon_id'])
    other_feature = pd.merge(other_feature, t4, on=['user_id', 'date_received'])
    other_feature = pd.merge(other_feature, t5, on=['user_id', 'coupon_id', 'date_received'])
    other_feature = pd.merge(other_feature, t7, on=['user_id', 'coupon_id', 'date_received'])
    return other_feature


def coupon_feature(dataset):
    dataset['day_of_week'] = dataset.date_received.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)    #优惠券领取日期是一周的第几天
    dataset['day_of_month'] = dataset.date_received.astype('str').apply(lambda x: int(x[6:8]))  #优惠券领取日期是一月的第几天
    dataset['days_distance'] = dataset.date_received.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 6, 30)).days)   #优惠券领取日期距离6月30号多少天
    dataset['discount_man'] = dataset.discount_rate.apply(get_discount_man)     #优惠券满多少元
    dataset['discount_jian'] = dataset.discount_rate.apply(get_discount_jian)   #优惠券减多少元
    dataset['is_man_jian'] = dataset.discount_rate.apply(is_man_jian)   #优惠类型是否为满减，若是，则为1，不是，则为0
    dataset['discount_rate'] = dataset.discount_rate.apply(calc_discount_rate)  #优惠券折扣率
    d = dataset[['coupon_id']]
    d['coupon_count'] = 1
    d = d.groupby('coupon_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, d, on='coupon_id', how='left')
    return dataset


def merchant_feature(feature):
    merchant = feature[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]

    t = merchant[['merchant_id']]
    t.drop_duplicates(inplace=True)

    t1 = merchant[merchant.date != 'null'][['merchant_id']]
    t1['total_sales'] = 1
    t1 = t1.groupby('merchant_id').agg('sum').reset_index()

    t2 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id']]  #用优惠消费的日期，即正样本
    t2['sales_use_coupon'] = 1
    t2 = t2.groupby('merchant_id').agg('sum').reset_index()

    t3 = merchant[merchant.coupon_id != 'null'][['merchant_id']]
    t3['total_coupon'] = 1
    t3 = t3.groupby('merchant_id').agg('sum').reset_index()

    t4 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id', 'distance']]
    t4.replace('null', -1, inplace=True)
    t4.distance = t4.distance.astype('int')
    t4.replace(-1, np.nan, inplace=True)
    t5 = t4.groupby('merchant_id').agg('min').reset_index()
    t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

    t6 = t4.groupby('merchant_id').agg('max').reset_index()
    t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

    t7 = t4.groupby('merchant_id').agg('mean').reset_index()
    t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

    t8 = t4.groupby('merchant_id').agg('median').reset_index()
    t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

    merchant_feature = pd.merge(t, t1, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t2, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t3, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t5, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t6, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t7, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t8, on='merchant_id', how='left')
    merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
    merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_use_coupon.astype(
        'float') / merchant_feature.total_coupon    #商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
    merchant_feature['coupon_rate'] = merchant_feature.sales_use_coupon.astype(
        'float') / merchant_feature.total_sales     #用券购买率，即用券购买的数量/购买的总量
    merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
    return merchant_feature


def user_feature(feature):
    user = feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

    t = user[['user_id']]
    t.drop_duplicates(inplace=True)

    t1 = user[user.date != 'null'][['user_id', 'merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1.merchant_id = 1
    t1 = t1.groupby('user_id').agg('sum').reset_index()
    t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

    t2 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id', 'distance']]
    t2.replace('null', -1, inplace=True)
    t2.distance = t2.distance.astype('int')
    t2.replace(-1, np.nan, inplace=True)
    t3 = t2.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

    t4 = t2.groupby('user_id').agg('max').reset_index()
    t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

    t5 = t2.groupby('user_id').agg('mean').reset_index()
    t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

    t6 = t2.groupby('user_id').agg('median').reset_index()
    t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

    t7 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('user_id').agg('sum').reset_index()

    t8 = user[user.date != 'null'][['user_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('user_id').agg('sum').reset_index()

    t9 = user[user.coupon_id != 'null'][['user_id']]
    t9['coupon_received'] = 1
    t9 = t9.groupby('user_id').agg('sum').reset_index()

    t10 = user[(user.date_received != 'null') & (user.date != 'null')][['user_id', 'date_received', 'date']]
    t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
    t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    t10 = t10[['user_id', 'user_date_datereceived_gap']]

    t11 = t10.groupby('user_id').agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    t12 = t10.groupby('user_id').agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    t13 = t10.groupby('user_id').agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    user_feature = pd.merge(t, t1, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t3, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t4, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t5, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t6, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t7, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t8, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t9, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t11, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t12, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t13, on='user_id', how='left')
    user_feature.count_merchant = user_feature.count_merchant.replace(np.nan, 0)
    user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace(np.nan, 0)
    user_feature['buy_use_coupon_rate'] = user_feature.buy_use_coupon.astype(
        'float') / user_feature.buy_total.astype(
        'float')    #用户用优惠券购买次数/用户所有的购买次数
    user_feature['user_coupon_transfer_rate'] = user_feature.buy_use_coupon.astype(
        'float') / user_feature.coupon_received.astype('float')     #用户领取优惠券后进行核销率
    user_feature.buy_total = user_feature.buy_total.replace(np.nan, 0)
    user_feature.coupon_received = user_feature.coupon_received.replace(np.nan, 0)
    return user_feature


def user_merchant(feature):
    all_user_merchant = feature[['user_id', 'merchant_id']]
    all_user_merchant.drop_duplicates(inplace=True)

    t = feature[['user_id', 'merchant_id', 'date']]
    t = t[t.date != 'null'][['user_id', 'merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t.drop_duplicates(inplace=True)

    t1 = feature[['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    t2 = feature[['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    t3 = feature[['user_id', 'merchant_id']]
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)

    t4 = feature[['user_id', 'merchant_id', 'date', 'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)

    user_merchant = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t1, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t2, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t3, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t4, on=['user_id', 'merchant_id'], how='left')
    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan, 0)
    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_received.astype('float')     #用户领取商家的优惠券后核销率
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')
    user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype(
        'float') / user_merchant.user_merchant_any.astype('float')
    user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')
    return user_merchant


if __name__ == '__main__':
    off_train = pd.read_csv('data/ccf_offline_stage1_train.csv', header=None, keep_default_na=False)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    off_test = pd.read_csv('data/ccf_offline_stage1_test_revised.csv', header=None)
    off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
    on_train = pd.read_csv('data/ccf_online_stage1_train.csv', header=None)
    on_train.columns = ['user_id', 'merchant_id', 'action', 'coupon_id', 'discount_rate', 'date_received', 'date']

    dataset3 = off_test
    feature3 = off_train[((off_train.date >= '20160315') & (off_train.date <= '20160630')) | (
            (off_train.date == 'null') & (off_train.date_received >= '20160315') & (
            off_train.date_received <= '20160630'))]
    dataset2 = off_train[(off_train.date_received >= '20160515') & (off_train.date_received <= '20160615')]
    feature2 = off_train[(off_train.date >= '20160201') & (off_train.date <= '20160514') | (
            (off_train.date == 'null') & (off_train.date_received >= '20160201') & (
            off_train.date_received <= '20160514'))]
    dataset1 = off_train[(off_train.date_received >= '20160414') & (off_train.date_received <= '20160514')]
    feature1 = off_train[(off_train.date >= '20160101') & (off_train.date <= '20160413') | (
            (off_train.date == 'null') & (off_train.date_received >= '20160101') & (
            off_train.date_received <= '20160413'))]

    other_feature3 = other_feature(dataset3)
    other_feature3.to_csv('data/other_feature3.csv', index=None)
    other_feature2 = other_feature(dataset2)
    other_feature2.to_csv('data/other_feature2.csv', index=None)
    other_feature1 = other_feature(dataset1)
    other_feature1.to_csv('data/other_feature1.csv', index=None)
    dataset3 = coupon_feature(dataset3)
    dataset3.to_csv('data/coupon3_feature.csv', index=None)
    dataset2 = coupon_feature(dataset2)
    dataset2.to_csv('data/coupon2_feature.csv', index=None)
    dataset1 = coupon_feature(dataset1)
    dataset1.to_csv('data/coupon1_feature.csv', index=None)
    merchant_feature3 = merchant_feature(feature3)
    merchant_feature3.to_csv('data/merchant3_feature.csv', index=None)
    merchant_feature2 = merchant_feature(feature2)
    merchant_feature2.to_csv('data/merchant2_feature.csv', index=None)
    merchant_feature1 = merchant_feature(feature1)
    merchant_feature1.to_csv('data/merchant1_feature.csv', index=None)
    user3_feature = user_feature(feature3)
    user3_feature.to_csv('data/user3_feature.csv', index=None)
    user2_feature = user_feature(feature2)
    user2_feature.to_csv('data/user2_feature.csv', index=None)
    user1_feature = user_feature(feature1)
    user1_feature.to_csv('data/user1_feature.csv', index=None)
    user_merchant3 = user_merchant(feature3)
    user_merchant3.to_csv('data/user_merchant3.csv', index=None)
    user_merchant2 = user_merchant(feature2)
    user_merchant2.to_csv('data/user_merchant2.csv', index=None)
    user_merchant1 = user_merchant(feature1)
    user_merchant1.to_csv('data/user_merchant1.csv', index=None)

    coupon3 = pd.read_csv('data/coupon3_feature.csv')
    merchant3 = pd.read_csv('data/merchant3_feature.csv')
    user3 = pd.read_csv('data/user3_feature.csv')
    user_merchant3 = pd.read_csv('data/user_merchant3.csv')
    other_feature3 = pd.read_csv('data/other_feature3.csv')
    dataset3 = pd.merge(coupon3, merchant3, on='merchant_id', how='left')
    dataset3 = pd.merge(dataset3, user3, on='user_id', how='left')
    dataset3 = pd.merge(dataset3, user_merchant3, on=['user_id', 'merchant_id'], how='left')
    dataset3 = pd.merge(dataset3, other_feature3, on=['user_id', 'coupon_id', 'date_received'], how='left')
    dataset3.drop_duplicates(inplace=True)

    dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan, 0)
    dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan, 0)
    dataset3.user_merchant_received = dataset3.user_merchant_received.replace(np.nan, 0)
    dataset3['is_weekend'] = dataset3.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(dataset3.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    dataset3 = pd.concat([dataset3, weekday_dummies], axis=1)
    dataset3.drop(['merchant_id', 'day_of_week', 'coupon_count'], axis=1, inplace=True)
    dataset3 = dataset3.replace('null', np.nan)
    dataset3.to_csv('data/dataset3.csv', index=None)

    coupon2 = pd.read_csv('data/coupon2_feature.csv')
    merchant2 = pd.read_csv('data/merchant2_feature.csv')
    user2 = pd.read_csv('data/user2_feature.csv')
    user_merchant2 = pd.read_csv('data/user_merchant2.csv')
    other_feature2 = pd.read_csv('data/other_feature2.csv')
    dataset2 = pd.merge(coupon2, merchant2, on='merchant_id', how='left')
    dataset2 = pd.merge(dataset2, user2, on='user_id', how='left')
    dataset2 = pd.merge(dataset2, user_merchant2, on=['user_id', 'merchant_id'], how='left')
    dataset2 = pd.merge(dataset2, other_feature2, on=['user_id', 'coupon_id', 'date_received'], how='left')
    dataset2.drop_duplicates(inplace=True)

    dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan, 0)
    dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan, 0)
    dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan, 0)
    dataset2.date = dataset2.date.replace(np.nan, 0)
    dataset2.date_received = dataset2.date_received.replace(np.nan, 0)
    dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(dataset2.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    dataset2 = pd.concat([dataset2, weekday_dummies], axis=1)
    dataset2.date = dataset2.date.apply(covert_int)
    dataset2['label'] = dataset2.date.astype('str') + ':' + dataset2.date_received.astype('str')
    dataset2.label = dataset2.label.apply(get_label)
    dataset2.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1,
                  inplace=True)
    dataset2 = dataset2.replace('null', np.nan)
    dataset2.to_csv('data/dataset2.csv', index=None)

    coupon1 = pd.read_csv('data/coupon1_feature.csv')
    merchant1 = pd.read_csv('data/merchant1_feature.csv')
    user1 = pd.read_csv('data/user1_feature.csv')
    user_merchant1 = pd.read_csv('data/user_merchant1.csv')
    other_feature1 = pd.read_csv('data/other_feature1.csv')
    dataset1 = pd.merge(coupon1, merchant1, on='merchant_id', how='left')
    dataset1 = pd.merge(dataset1, user1, on='user_id', how='left')
    dataset1 = pd.merge(dataset1, user_merchant1, on=['user_id', 'merchant_id'], how='left')
    dataset1 = pd.merge(dataset1, other_feature1, on=['user_id', 'coupon_id', 'date_received'], how='left')
    dataset1.drop_duplicates(inplace=True)

    dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan, 0)
    dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan, 0)
    dataset1.user_merchant_received = dataset1.user_merchant_received.replace(np.nan, 0)
    dataset1.date = dataset1.date.replace(np.nan, 0)
    dataset1.date_received = dataset1.date_received.replace(np.nan, 0)
    dataset1['is_weekend'] = dataset1.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(dataset1.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    dataset1 = pd.concat([dataset1, weekday_dummies], axis=1)
    dataset1.date = dataset1.date.apply(covert_int)
    dataset1['label'] = dataset1.date.astype('str') + ':' + dataset1.date_received.astype('str')
    dataset1.label = dataset1.label.apply(get_label)
    dataset1.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1,
                  inplace=True)
    dataset1 = dataset1.replace('null', np.nan)
    dataset1.to_csv('data/dataset1.csv', index=None)
