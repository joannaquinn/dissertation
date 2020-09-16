import os
import datetime as dt
import csv
from chardet import detect
import random
import math
import numpy as np
import pandas as pd

from basic_imports import *
from misc_utils import *
from constants_jo import *
from load_files import *
from download_data import *

import matplotlib.pyplot as plt
import seaborn as sns

def cut_df(df, start, end, timestamp_col = 'timestamp'):
    '''
    given a start and end time as timestamps, 
    returns sliced dataframe
    '''
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    new_df = df[df[timestamp_col] < str(end)]
    new_df = new_df[new_df[timestamp_col] >= str(start)]
    return new_df

def hour_from_timestamp(ts):
    return pd.Timestamp(ts).hour

def weekday_from_timestamp(ts):
    # monday = 0, sunday = 6
    return pd.Timestamp(ts).weekday()

def time_from_timestamp(ts):
    '''
    create generalised time value from timestamp
    '''
    ts = pd.Timestamp(ts)
    return pd.Timestamp(year=2001, month=1, day=1, hour=ts.hour, minute=ts.minute, second=ts.second)

def dates_in_timestamp(dates, ts):
    ts = pd.Timestamp(ts)
    for date in dates:
        if date == ts.date():
            return True
    return False

def date_from_timestamp(ts):
    return pd.Timestamp(ts).date()

def weekday_only_df(df, timestamp_col = 'timestamp'):
    df2 = df.copy()
    df2['weekday'] = df2.apply(lambda x: weekday_from_timestamp(x[timestamp_col]), axis=1)
    return df2[df2['weekday'] < 5]

def weekend_only_df(df, timestamp_col='timestamp'):
    df2 = df.copy()
    df2['weekday'] = df2.apply(lambda x: weekday_from_timestamp(x[timestamp_col]), axis=1)
    return df2[df2['weekday'] >= 5]

def combine_dfs(dfs):
    return pd.concat(dfs)

def new_date_col(date, all_dates, new_dates):
    try:
        i = all_dates.index(date)
        return new_dates[i]
    except:
        return float("nan")

def new_timestamp_col(date, ts):
    ts = pd.Timestamp(ts)
    return pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=ts.hour, minute=ts.minute, second=ts.second)

def plot_hist_days(df, filename=None, col='latitude', figsize=(15,10), timestamp_col='timestamp'):
    '''
    plot distribution of the data collected on each day
    '''
    hist_df = df.set_index(pd.DatetimeIndex(df[timestamp_col]), drop=False, inplace=False)
    hist_df = hist_df[[col]].groupby(pd.Grouper(freq='D')).count()
    
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    plot = sns.barplot(x=hist_df.index, y=hist_df[col], ax=ax);
    ax.set_xlabel('Timestamp');
    ax.set_ylabel('Count');
    ax.xaxis_date();
    ax.xaxis.set_major_formatter(plt.FixedFormatter(hist_df.index.to_series().dt.strftime("%d %b")));
    for tick in ax.get_xticklabels():
        tick.set_rotation(65);
    
    if filename is not None:
        fig = plot.get_figure()
        fig.savefig(filename)
    
def plot_hist_times(df, filename=None, col='latitude', figsize=(15,10), timestamp_col='timestamp'):
    '''
    plot distribution of data collected for each time period (hour)
    '''
    hist_df = df.set_index(pd.DatetimeIndex(df[timestamp_col]), drop=False, inplace=False)
    hist_df['hour'] = hist_df.apply(lambda x: hour_from_timestamp(x[timestamp_col]), axis=1)
    hist_df = hist_df[[col]].groupby(hist_df['hour']).count()
    
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    plot = sns.barplot(x=hist_df.index, y=hist_df[col], ax=ax);
    ax.set_xlabel('Hour');
    ax.set_ylabel('Count');
    ax.xaxis_date();
    
    if filename is not None:
        fig = plot.get_figure()
        fig.savefig(filename)
    
def plot_hist_weekdays(df, filename=None, col='latitude', figsize=(15,10), timestamp_col='timestamp'):
    '''
    plot distribution of data collected for each day of the week
    note mon = 0, tues = 1, ... , sun = 6
    '''
    hist_df = df.set_index(pd.DatetimeIndex(df[timestamp_col]), drop=False, inplace=False)
    hist_df['weekday'] = hist_df.apply(lambda x: weekday_from_timestamp(x[timestamp_col]), axis=1)
    hist_df = hist_df[[col]].groupby(hist_df['weekday']).count()
    
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    plot = sns.barplot(x=hist_df.index, y=hist_df[col], ax=ax);
    ax.set_xlabel('Day of the week');
    ax.set_ylabel('Count');
    ax.xaxis_date();
    
    if filename is not None:
        fig = plot.get_figure()
        fig.savefig(filename)
    
def plot_trend_all(df, filename=None, interval=15, pms=['pm1','pm2_5','pm10'], figsize=(15,10), timestamp_col='timestamp'):
    '''
    plot PM trends over the full data collection period of the dataset
    '''
    df2 = df.set_index(pd.DatetimeIndex(df[timestamp_col]), drop=False, inplace=False)
    df2 = df2.resample(f'{interval}Min').mean()
    df2['timestamp2'] = df2.index.copy()
    data=pd.melt(df2, id_vars = 'timestamp2', value_vars=pms)
    
    fig, ax = plt.subplots(figsize=figsize)
    plot = sns.lineplot(x='timestamp2',y='value',hue='variable', data=data,ax=ax)
    
    if filename is not None:
        fig = plot.get_figure()
        fig.savefig(filename)
    
def plot_trend_times(df, filename=None, interval=15, pms=['pm1','pm2_5','pm10'], figsize=(15,10), timestamp_col='timestamp'):
    '''
    plot PM trends over time, averaging over all days in the dataset
    '''
    df2 = df.copy()
    df2['time'] = df.apply(lambda x: time_from_timestamp(x[timestamp_col]), axis=1)
    df2 = df2.set_index(pd.DatetimeIndex(df2['time']), drop=False, inplace=False)
    df2 = df2.resample(f'{interval}Min').mean()
    df2['time2'] = df2.index.copy()
    data=pd.melt(df2, id_vars = 'time2', value_vars=pms)

    fig, ax = plt.subplots(figsize=(20,15))
    plot = sns.lineplot(x='time2',y='value',hue='variable', data=data,ax=ax)
    
    if filename is not None:
        fig = plot.get_figure()
        fig.savefig(filename)
    
def plot_all(project_name, suffix=''):
    '''
    wrapper for plot functions
    '''
    data_dir = project_mapping[project_name][2]
    plot_dir = project_mapping[project_name][3]
    
    labels = ('personal','static')
    for label in labels:
        data_path = f'{data_dir}{project_name}_{label}_all_rawtime.csv'
        df = pd.read_csv(data_path)
        
        plot_hist_days(df, f'{plot_dir}{project_name}_{label}_all_rawtime_hist_days{suffix}.png')
        plot_hist_times(df, f'{plot_dir}{project_name}_{label}_all_rawtime_hist_times{suffix}.png')
        plot_hist_weekdays(df, f'{plot_dir}{project_name}_{label}_all_rawtime_hist_weekdays{suffix}.png')
        
        plot_trend_all(df, f'{plot_dir}{project_name}_{label}_all_rawtime_trend_days{suffix}.png')
        plot_trend_times(df, f'{plot_dir}{project_name}_{label}_all_rawtime_trend_times{suffix}.png')
    
def create_df_average_day(df, start=None, end=None, dates=None, timestamp_col='timestamp'):
    '''
    creates dataframe which combines data from all valid days 
    if not specified, uses all days in dataframe
    if start and end time specified, uses slice of data
    if specific dates specified (as list of dt.date objects), only uses those dates
    '''
    df2 = df.copy()
    df2['time'] = df2.apply(lambda x: time_from_timestamp(x[timestamp_col]), axis=1)
    df2['date'] = df2.apply(lambda x: date_from_timestamp(x[timestamp_col]), axis=1)
    df2 = df2.set_index(pd.DatetimeIndex(df2['time']), drop=False, inplace=False)
    
    if dates is not None:
        df2['include'] = df2.apply(lambda x: dates_in_timestamp(dates, x[timestamp_col]), axis=1)
        df2 = df2[df2['include'] == True]
        df2 = df2.drop(columns=['include'])
        return df2
    
    elif start is not None and end is not None:
        return cut_df(df2, start, end, timestamp_col = timestamp_col)
    
    else:
        return df2
    
def create_df_average_time(df, start_time=dt.time(0,0,0), end_time=dt.time(23,59,59), start=None, end=None, dates=None, timestamp_col='timestamp'):
    '''
    creates dataframe which combines data from all valid days 
    start and end times are datetimes which form the time slice for each day
    if not specified, uses all days in dataframe
    if specific dates specified (as list of dt.date objects), only uses those dates
    '''
    df2 = df.copy()
    df2['time'] = df2.apply(lambda x: time_from_timestamp(x[timestamp_col]), axis=1)
    df2['date'] = df2.apply(lambda x: date_from_timestamp(x[timestamp_col]), axis=1)
    
    # take subset of data that falls within start_time and end_time
    start_time = dt.datetime.combine(dt.date(2001,1,1),start_time)
    end_time = dt.datetime.combine(dt.date(2001,1,1),end_time)
    df2 = df2.set_index(pd.DatetimeIndex(df2['time']), drop=False, inplace=False)
    df2 = cut_df(df2, pd.Timestamp(start_time), pd.Timestamp(end_time), timestamp_col='time')
    
    df2 = df2.set_index(pd.DatetimeIndex(df2['date']))
    
    if dates is not None:
        df2['include'] = df2.apply(lambda x: dates_in_timestamp(dates, x[timestamp_col]), axis=1)
        df2 = df2[df2['include'] == True]
        df2 = df2.drop(columns=['include'])
        return df2
    
    elif start is not None and end is not None:
        return cut_df(df2, start, end, timestamp_col = timestamp_col)
    
    else:
        return df2
    
def create_df_concurrent_days(static_df, personal_df, start_time=dt.time(0,0,0), end_time=dt.time(23,59,59), timestamp_col='timestamp'):
    '''
    creates static and personal dataframe where
    gaps between days are removed so that days are concurrent
    new da
    '''
    static_df = static_df.copy()
    personal_df = personal_df.copy()
    static_df['date'] = static_df.apply(lambda x: date_from_timestamp(x[timestamp_col]), axis=1)
    personal_df['date'] = personal_df.apply(lambda x: date_from_timestamp(x[timestamp_col]), axis=1)
    static_df['time'] = static_df.apply(lambda x: time_from_timestamp(x[timestamp_col]), axis=1)
    personal_df['time'] = personal_df.apply(lambda x: time_from_timestamp(x[timestamp_col]), axis=1)
    
    start_time = dt.datetime.combine(dt.date(2001,1,1),start_time)
    end_time = dt.datetime.combine(dt.date(2001,1,1),end_time)        
    static_df = cut_df(static_df, pd.Timestamp(start_time), pd.Timestamp(end_time), timestamp_col='time')
    personal_df = cut_df(personal_df, pd.Timestamp(start_time), pd.Timestamp(end_time), timestamp_col='time')
    
    static_unique_dates = list(static_df['date'].unique())
    personal_unique_dates = list(personal_df['date'].unique())
    all_dates = sorted([x for x in static_unique_dates if x in personal_unique_dates])
    print(f'Static df has {len(static_unique_dates)} dates, personal df has {len(personal_unique_dates)} dates')
    print(f'New dfs have {len(all_dates)} dates')
    new_dates = [dt.date(2001,1,1)+dt.timedelta(days=i) for i in range(len(all_dates))]
    
    static_df['new_date'] = static_df.apply(lambda x: new_date_col(x['date'], all_dates, new_dates), axis=1)
    personal_df['new_date'] = personal_df.apply(lambda x: new_date_col(x['date'], all_dates, new_dates), axis=1)
    static_df = static_df.dropna()
    personal_df = personal_df.dropna()
    
    static_df['new_timestamp'] = static_df.apply(lambda x: new_timestamp_col(x['new_date'],x[timestamp_col]), axis=1)
    personal_df['new_timestamp'] = personal_df.apply(lambda x: new_timestamp_col(x['new_date'],x[timestamp_col]), axis=1)
    print('finished processing df; new date column is "new_timestamp"')
    
    return static_df, personal_df