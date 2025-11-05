import ast
import json
import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

HEROES_DICT_FILE = "../docs/heroes_data.json"
INPUT_RAW_DATA_DIR = "../data/raw"
OUTPUT_PROCESSED_DIR = '../data/processed'
OUTPUT_TRAIN_DATA_DIR = os.path.join(OUTPUT_PROCESSED_DIR, 'train.parquet')
OUTPUT_VAL_DATA_DIR = os.path.join(OUTPUT_PROCESSED_DIR, 'validation.parquet')
OUTPUT_TEST_DATA_DIR = os.path.join(OUTPUT_PROCESSED_DIR, 'test.parquet')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

heroes_data = {}
with open(HEROES_DICT_FILE, "r") as file:
    full_data = json.load(file)
    for item in full_data:
        heroes_data[item['id']] = item


def print_dataframe_info(label: str, dataframe: pd.DataFrame):
    win_counts = dataframe['radiant_win'].value_counts()
    logging.info("{}:\n\tTotal records: {}\n\tRadiant win records: {}\n\tDire win records: {}"
                 .format(label, len(dataframe), win_counts[1], win_counts[0]))


allow_game_types = ['1', '2', '16', '22']
allowed_columns = [
    'match_id',
    'radiant_win',
    'game_mode',
    'avg_rank_tier',
    'duration',
    'radiant_team',
    'dire_team']
columns_to_normalize = ['avg_rank_tier', 'duration']
dataframes = []

csv_files = os.listdir(INPUT_RAW_DATA_DIR)

for file in tqdm(csv_files, desc='Reading CSV files'):
    df = pd.read_csv(os.path.join(INPUT_RAW_DATA_DIR, file))

    # filter by game mode
    df = df[df['game_mode'].astype(str).isin(allow_game_types)]

    # filter by duration
    df = df.loc[(df['duration'] >= 1200) & (df['duration'] <= 6000)]
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
logging.info('Combined dataframe shape: {}'.format(combined_df.shape))

# filter by columns
combined_df = combined_df[allowed_columns]
logging.info('Filtered dataframe shape: {}'.format(combined_df.shape))

# change radiant_win to int
combined_df['radiant_win'] = combined_df['radiant_win'].astype(str).map({'True': 1, 'False': 0})
logging.info('Changed radiant_win column to int')

# change radiant_team and dire_team to list
combined_df['radiant_team'] = combined_df['radiant_team'].apply(lambda s: ast.literal_eval(s))
logging.info('Changed radiant_team column to list')
combined_df['dire_team'] = combined_df['dire_team'].apply(lambda s: ast.literal_eval(s))
logging.info('Changed dire_team column to list')

# check teams length and unique
combined_df = combined_df[combined_df.apply(lambda row:
                                            len(row['radiant_team']) == 5 and
                                            len(row['dire_team']) == 5 and
                                            len(set(row['radiant_team'] + row['dire_team'])) == 10,
                                            axis=1)]
logging.info('Filtered dataframe by teams length and unique: {}'.format(combined_df.shape))

# add heroes primary_attributes
combined_df['radiant_primary_attributes'] = combined_df['radiant_team'].apply(
    lambda team: [heroes_data[hero]['primary_attr'] for hero in team])
logging.info('Added to radiant heroes primary_attributes')
combined_df['dire_primary_attributes'] = combined_df['dire_team'].apply(
    lambda team: [heroes_data[hero]['primary_attr'] for hero in team])
logging.info('Added to dire heroes primary_attributes')

# add heroes attack_type
combined_df['radiant_attack_type'] = combined_df['radiant_team'].apply(
    lambda team: [heroes_data[hero]['attack_type'] for hero in team])
logging.info('Added to radiant heroes attack_type')
combined_df['dire_attack_type'] = combined_df['dire_team'].apply(
    lambda team: [heroes_data[hero]['attack_type'] for hero in team])
logging.info('Added to dire heroes attack_type')

# add heroes roles
combined_df['radiant_roles'] = combined_df['radiant_team'].apply(
    lambda team: [heroes_data[hero]['roles'] for hero in team])
logging.info('Added to radiant heroes roles')
combined_df['dire_roles'] = combined_df['dire_team'].apply(lambda team: [heroes_data[hero]['roles'] for hero in team])
logging.info('Added to dire heroes roles')

# add heroes winrate
combined_df['radiant_winrate'] = combined_df['radiant_team'].apply(
    lambda team: [heroes_data[hero]['winrate_scaled'] for hero in team])
logging.info('Added to radiant heroes winrate')
combined_df['dire_winrate'] = combined_df['dire_team'].apply(
    lambda team: [heroes_data[hero]['winrate_scaled'] for hero in team])
logging.info('Added to dire heroes winrate')

# add heroes pickrate
combined_df['radiant_pickrate'] = combined_df['radiant_team'].apply(
    lambda team: [heroes_data[hero]['pickrate_scaled'] for hero in team])
logging.info('Added to radiant heroes pickrate')
combined_df['dire_pickrate'] = combined_df['dire_team'].apply(
    lambda team: [heroes_data[hero]['pickrate_scaled'] for hero in team])
logging.info('Added to dire heroes pickrate')

# replace hero ids to indexes
combined_df['radiant_team'] = combined_df['radiant_team'].apply(
    lambda team: [heroes_data[hero]['index'] for hero in team])
logging.info('Replaced radiant heroes ids to indexes')
combined_df['dire_team'] = combined_df['dire_team'].apply(lambda team: [heroes_data[hero]['index'] for hero in team])
logging.info('Replaced dire heroes ids to indexes')

# normalize
scaler = MinMaxScaler()
combined_df[columns_to_normalize] = scaler.fit_transform(combined_df[columns_to_normalize])
logging.info('Columns {} normalized'.format(columns_to_normalize))

print_dataframe_info("Dataset before balancing", combined_df)

# balance by radiant win
radiant_df = combined_df[combined_df['radiant_win'] == 1]
dire_df = combined_df[combined_df['radiant_win'] == 0]

min_size = min(len(radiant_df), len(dire_df))

radiant_sampled = radiant_df.sample(n=min_size, random_state=42)
dire_sampled = dire_df.sample(n=min_size, random_state=42)

balanced_df = pd.concat([radiant_sampled, dire_sampled], ignore_index=True)

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print_dataframe_info("Dataset after balancing", balanced_df)

# split into train, validation, test
train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.3,
    stratify=balanced_df['radiant_win'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['radiant_win'],
    random_state=42
)

print_dataframe_info("Train data", train_df)
print_dataframe_info("Validation data", val_df)
print_dataframe_info("Test data", test_df)
train_df.to_parquet(OUTPUT_TRAIN_DATA_DIR, index=False)
logging.info('Train data saved to {}'.format(OUTPUT_TRAIN_DATA_DIR))
val_df.to_parquet(OUTPUT_VAL_DATA_DIR, index=False)
logging.info('Validation data saved to {}'.format(OUTPUT_VAL_DATA_DIR))
test_df.to_parquet(OUTPUT_TEST_DATA_DIR, index=False)
logging.info('Test data saved to {}'.format(OUTPUT_TEST_DATA_DIR))
