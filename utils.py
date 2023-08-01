# import config as cfg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import Counter

def generate_playInSusMapRate(df_validation, df_timediff, sus_map_index):
    total_map_count = df_timediff.groupby('ChaNum')['mapId'].count()
    sus_map_count = df_timediff[df_timediff['mapId'].isin(sus_map_index)].groupby('ChaNum')['mapId'].count()
    play_in_sus_map_rate = (sus_map_count / total_map_count).round(3)
    df_validation['playInSusMapRate'] = df_validation['ChaNum'].map(play_in_sus_map_rate).fillna(0)
    # ensure value is between 0 and 1
    df_validation['playInSusMapRate'] = df_validation['playInSusMapRate'].clip(0, 1)
    return df_validation

def generate_maxTimesInOneDay(df_validation, df_timediff):
    df_timediff['CreateTime'] = pd.to_datetime(df_timediff['CreateTime'])
    df_timediff['CreateTime'] = df_timediff['CreateTime'].dt.date
    def assign_maxTimesInOneDay(row):
        df_cha = df_timediff[df_timediff['ChaNum'] == row['ChaNum']]
        daily_counts = df_cha.groupby('CreateTime').size()
        return daily_counts.max()
    df_validation['maxTimesInOneDay'] = df_validation.apply(assign_maxTimesInOneDay, axis=1)
    return df_validation

def generate_maxRepeatedPattern(df_validation, df_timediff):
    def get_seq_timediff(ChaNum):
        df = df_timediff[df_timediff['ChaNum'] == ChaNum].sort_values('id') # rec_id change to id
        return df.timediff.values.tolist()

    def find_max_repeated_pattern(sequence):
        # convert to string
        str_seq = ' '.join(map(str, sequence))
        # find all repeated pattern
        patterns = re.findall(r'(\b\d+\s+\d+\b)(?=\s+\1)', str_seq)
        # count appear times of each patterns
        count = Counter(patterns)
        if count:
            # found most frequence pattern and count
            _, max_count = count.most_common(1)[0]
        else:
            max_count = 0
        # only return  max count value
        return max_count

    def assign_maxRepeatedPattern(row):
        chanum = row['ChaNum']
        seq = get_seq_timediff(chanum)
        return find_max_repeated_pattern(seq)
    df_validation['maxRepeatedPattern'] = df_validation.apply(assign_maxRepeatedPattern, axis=1)
    return df_validation

def generate_maxNonRepeatedPattern(df_validation, df_timediff):
    def get_seq_timediff(ChaNum):
        df = df_timediff[df_timediff['ChaNum'] == ChaNum].sort_values('id') # rec_id change to id
        return df.timediff.values.tolist()

    def find_max_non_repeated_pattern(sequence):
        # convert to string
        str_seq = ' '.join(map(str, sequence))
        # find all non-repeated pattern
        patterns = re.findall(r'(\b(\d+)\s+(?!\2)\d+\b)(?=\s+\1)', str_seq)
        # count appear times of each patterns
        count = Counter([pattern[0] for pattern in patterns])
        if count:
            # found most frequence pattern and count
            max_pattern, max_count = count.most_common(1)[0]
        else:
            max_count = 0
        # only return  max count value
        return max_count

    def assign_maxNonRepeatedPattern(row):
        chanum = row['ChaNum']
        seq = get_seq_timediff(chanum)
        return find_max_non_repeated_pattern(seq)
    df_validation['maxNonRepeatedPattern'] = df_validation.apply(assign_maxNonRepeatedPattern, axis=1)
    return df_validation

def generate_timediffAutocorr(df_validation, df_timediff):
    def get_seq_timediff(ChaNum):
        df = df_timediff[df_timediff['ChaNum'] == ChaNum].sort_values('id') # rec_id change to id
        return df.timediff.values.tolist()

    df_timediff.groupby('ChaNum')['timediff'].agg(['mean', 'std']).fillna(0)
    def assign_autocorr(row):
        seq = get_seq_timediff(row['ChaNum'])
        return round(pd.Series(seq).autocorr(), 3)
    df_validation['timediffAutocorr'] = df_validation.apply(assign_autocorr, axis=1).fillna(0)
    return df_validation

def generate_timediffMean(df_validation, df_timediff):
    def get_seq_timediff(ChaNum):
        df = df_timediff[df_timediff['ChaNum'] == ChaNum].sort_values('id') # rec_id change to id
        return df.timediff.values.tolist()

    def assign_mean(row):
        seq = get_seq_timediff(row['ChaNum'])
        return round(np.mean(seq), 3)

    df_validation['timediffMean'] = df_validation.apply(assign_mean, axis=1)
    return df_validation

def generate_timediffStd(df_validation, df_timediff):
    def get_seq_timediff(ChaNum):
        df = df_timediff[df_timediff['ChaNum'] == ChaNum].sort_values('id') # rec_id change to id
        return df.timediff.values.tolist()

    def assign_std(row):
        seq = get_seq_timediff(row['ChaNum'])
        return round(np.std(seq), 3)

    df_validation['timediffStd'] = df_validation.apply(assign_std, axis=1)
    return df_validation

def generate_idxmaxTimediffCount(df_validation, df_timediff):
    def assign_idxmaxTimediffCount(row):
        df = df_timediff[(df_timediff['ChaNum'] == row['ChaNum'])].sort_values('id')
        # value count
        vc = df['timediff'].value_counts()
        return vc.idxmax()

    df_validation['idxmaxTimediffCount'] = df_validation.apply(assign_idxmaxTimediffCount, axis=1)
    return df_validation

def generate_maxTimediffCount(df_validation, df_timediff):
    def assign_maxTimediffCount(row):
        df = df_timediff[(df_timediff['ChaNum'] == row['ChaNum'])].sort_values('id')
        # value count
        vc = df['timediff'].value_counts()
        return vc.max()

    df_validation['maxTimediffCount'] = df_validation.apply(assign_maxTimediffCount, axis=1)
    return df_validation

def generate_timediffCountdiff(df_validation, df_timediff):
    def assign_timediffCountdiff(row):
        df = df_timediff[(df_timediff['ChaNum'] == row['ChaNum'])].sort_values('id')
        # value count
        vc = df['timediff'].value_counts()
        # frequence
        if len(vc) < 2:
            diff = 0
        elif len(vc) < 3:
            top_two_counts = vc.nlargest(2)
            diff = top_two_counts.iloc[0] - top_two_counts.iloc[1] # top1 - top2
        else:
            top_three_counts = vc.nlargest(3)
            diff = top_three_counts.iloc[0] - top_three_counts.iloc[2] # top1 - top3

        return diff

    df_validation['timediffCountdiff'] = df_validation.apply(assign_timediffCountdiff, axis=1)
    return df_validation

def saveTimediffPlot(df_cha, ChaNum, folder):
    plt.cla()
    df_cha = df_cha.sort_values('id')
    grouped = df_cha.groupby('ChaNum')
    for name, group in grouped:
        plt.plot(group['CreateTime'], group['timediff'], label=name)

    # setup plot format
    plt.xlabel('Date')
    plt.ylabel('timediff')
    plt.xticks(rotation=90)

    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    os.makedirs(folder, exist_ok=True)

    # save plot image
    image_path_name = folder + "/" + str(ChaNum) + ".png"
    plt.savefig(image_path_name)