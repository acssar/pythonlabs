import pandas as pd
import numpy as np


def df_sex_filter(df):
    """
    reads data frame
    convert female values to 0
    convert male values to 1
    eliminate empty values
    Parameters
    ----------
    df : pandas.DataFrame
    Returns
    -------
    df : pandas.DataFrame
    """
    df['sex'] = df['sex'].str.lower()
    not_stated = ['не указан', '-']
    df = df[~df['sex'].isin(not_stated)].copy()
    df['sex'] = (~df['sex'].str.startswith('ж')).astype(int)
    return df


def df_row_number_nan_filter(df):
    """
    convert NaN values to max value in col
    Parameters
    ----------
    df : pandas.DataFrame
    Returns
    -------
    df : pandas.DataFrame
    """
    m = df['row_number'].max()
    df['row_number'] = df['row_number'].fillna(m)
    return df


def df_drink_amount_filter(df, min=0, max=10):
    """                                             # TODO: rename min max
    destroys too large and negative amount of drinks
    Parameters
    ----------
    df : pandas.DataFrame
    min: int
     minimum of drinks
    max: int
     maximum of drinks
    Returns
    -------
    df : pandas.DataFrame
    """
    s = df['liters_drunk'].between(min, max)
    med = round(df.loc[s, 'liters_drunk'].mean(), 2)
    df.loc[~s, 'liters_drunk'] = med
    return df


def df_alcohol_filter(df):
    """
    alcohol drink to 1
    not alcohol drink to 0
    Parameters
    ----------
    df : pandas.DataFrame
    Returns
    -------
    df : pandas.DataFrame
    """
    df['drink'] = (df['drink'].str.lower().str.contains('пиво|beer')).astype(int)
    return df


def df_age_filter(df, y=18, o=50):
    """
    split column 'session_start' to 3 columns: 'morning', 'afternoon' and 'evening'
    on 3 intervals: [0, y] , [y; o] , [o; 1000]
    Parameters
    ----------
    df : pandas.DataFrame
    y : int
     bound of young
    o : int
     bound of old
    Returns
    -------
    df : pandas.DataFrame
    """
    df['age_young'] = pd.cut(df['age'], labels=['young'], bins=[0, y])
    df['age_medium'] = pd.cut(df['age'], labels=['medium'], bins=[y, o])
    df['age_old'] = pd.cut(df['age'], labels=['old'], bins=[o, np.inf])
    df = df.drop(columns=['age'])
    return df


def df_checks_filter_special(df1, df2_file_name='cinema_sessions.csv', start=0, afternoon=12, evening=18, end=24):
    """
    split column 'session_start' to 3 columns: 'morning', 'afternoon' and 'evening'
    on 3 intervals: [start; afternoon-1] , [afternoon; evening-1] , [evening; end]

    convert 'session_start' column to datetime format

    Parameters
    ----------
    df1 : pandas.DataFrame
    df2_file_name='cinema_sessions.csv' : string
    start : int
    afternoon : int
    evening : int
    end : int
    Returns
    -------
    df1 : pandas.DataFrame
    """
    df2 = pd.read_csv(df2_file_name, delimiter=' ')
    df2['session_start'] = pd.to_datetime(df2['session_start'])
    df1['morning1'] = df2['session_start'].dt.hour.between(start, afternoon - 1).astype(int)
    df1['afternoon1'] = df2['session_start'].dt.hour.between(afternoon, evening - 1).astype(int)
    df1['evening1'] = df2['session_start'].dt.hour.between(evening, end).astype(int)
    df1 = df1.merge(df2, on='check_number')
    df2.to_csv('oae.csv', sep=',')
    return df1


if __name__ == '__main__':
    data = pd.read_csv('titanic_with_labels.csv', delimiter=' ')
    data = df_sex_filter(data)
    data = df_row_number_nan_filter(data)
    data = df_drink_amount_filter(data)
    data = df_alcohol_filter(data)
    data = df_age_filter(data)
    df = df_checks_filter_special(data)

    data.to_csv("output.csv", sep=',')
    print(data)
