import pandas as pd
import numpy as np
import datetime
import os

def read_data(filepath):
    df = pd.read_csv(filepath)
    df.set_index('Date', inplace=True)
    return df

def normalization(df):
    ratio = df['Adj Close'] / df['Close']

    df['Open'] = df['Open'] * ratio
    df['High'] = df['High'] * ratio
    df['Low'] = df['Low'] * ratio
    df['Close'] = df['Close'] * ratio

    df.drop(columns=['Adj Close'], inplace=True)

def ind_max(x):
    a = np.array(x)
    a.sort()
    np.arga[-1]
    a[-2]


def labeling(df, windowSize):
    label_map = {'Hold': 0, 'Buy': 1, 'Sell': 2}
    roll = df['Close'].rolling(windowSize, center=True)

    df['Label'] = roll.apply(lambda x: label_map['Buy'] if np.argmin(x) == windowSize // 2 else (label_map['Sell'] if np.argmax(x) == windowSize // 2 else label_map['Hold']), raw=True)
    for label in label_map:
        df['Label'].replace(label_map[label], label, inplace=True)
    df.dropna(inplace=True)

    dates = pd.to_datetime(df.index)
    return df[(dates >= datetime.datetime(1996, 10, 31)) & (dates <= datetime.datetime(2017, 1, 1))]

if __name__ == "__main__":
    
    input_dir = './raw data'
    output_dir = './labeled data'

    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        files += [os.path.join(dirpath, f) for f in filenames]

    for f in files:
        data = read_data(f)
        normalization(data)
        data = labeling(data, 11)
        data.to_csv(output_dir+f.split(input_dir)[1])