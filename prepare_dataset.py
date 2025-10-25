import ast
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

hero_id_to_index = {}
with open("docs/heroes_dict.json", "r") as file:
    full_data = json.load(file)
    for item in full_data:
        hero_id_to_index[item['key']] = item['value']


def map_team(ids_line: str) -> list[int]:
    result = []
    ids = list(map(int, ast.literal_eval(ids_line)))
    for hero_id in ids:
        if hero_id not in hero_id_to_index:
            raise ValueError(f"Unknown hero id: {hero_id}")
        result.append(hero_id_to_index[hero_id])
    return result


def print_dataframe_info(label: str, dataframe: pd.DataFrame):
    print(f"{label}:\n\tTotal records: {len(dataframe)}")
    win_counts = dataframe['radiant_win'].value_counts()
    print(f"\tRadiant win records: {win_counts[1]}")
    print(f"\tDire win records: {win_counts[0]}")


dir_path = 'data/'
allow_game_types = ['1', '2', '16', '22']
allowed_columns = ['match_id', 'radiant_win', 'game_mode', 'avg_rank_tier', 'num_rank_tier', 'duration', 'radiant_team', 'dire_team']
columns_to_normalize = ['avg_rank_tier', 'num_rank_tier', 'duration']
dataframes = []

output_dir = 'datasets/'

csv_files = os.listdir(dir_path)

for file in tqdm(csv_files):
    df = pd.read_csv(os.path.join(dir_path, file))

    # filter by game mode
    df = df[df['game_mode'].astype(str).isin(allow_game_types)]

    # filter by columns
    df = df[allowed_columns]

    # change radiant_win to int
    df['radiant_win'] = df['radiant_win'].astype(str).map({'True': 1, 'False': 0})

    # heroes id to index
    df['radiant_team'] = df['radiant_team'].apply(lambda s: map_team(s))
    df['dire_team'] = df['dire_team'].apply(lambda s: map_team(s))

    # check teams length and unique
    df = df[df.apply(lambda row:
                     len(row['radiant_team']) == 5 and
                     len(row['dire_team']) == 5 and
                     len(row['radiant_team'] + row['dire_team']) == 10,
                     axis=1)]
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

scaler = MinMaxScaler()
combined_df[columns_to_normalize] = scaler.fit_transform(combined_df[columns_to_normalize])

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

train_df.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
val_df.to_parquet(os.path.join(output_dir, 'validation.parquet'), index=False)
test_df.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
