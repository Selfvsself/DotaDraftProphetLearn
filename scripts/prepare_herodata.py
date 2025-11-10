import json
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

INPUT_FILE = '../docs/hero_stats.json'
OUTPUT_FILE = '../docs/heroes_data.json'

primary_attr_dict = {
    "str": 1,
    "agi": 2,
    "int": 3,
    "all": 4
}
attack_type_dict = {
    "Melee": 1,
    "Ranged": 2
}
roles_dict = {
    "Carry": 1,
    "Support": 2,
    "Disabler": 3,
    "Durable": 4,
    "Escape": 5,
    "Initiator": 6,
    "Nuker": 7,
    "Pusher": 8
}

heroes_data = []
with open(INPUT_FILE, "r") as file:
    full_data = json.load(file)
logging.info(f'{INPUT_FILE} was loaded. Heroes num = {len(full_data)}')

for item in full_data:
    hero_id = item['id']
    hero_name = item['localized_name']
    hero_index = len(heroes_data) + 1
    primary_attr = item['primary_attr']
    if primary_attr not in primary_attr_dict:
        raise Exception('Unknown primary_attr: ' + primary_attr)
    primary_attr = primary_attr_dict[primary_attr]

    attack_type = item['attack_type']
    if attack_type not in attack_type_dict:
        raise Exception('Unknown attack_type: ' + attack_type)
    attack_type = attack_type_dict[attack_type]
    roles = item['roles']
    roles = [roles_dict[role] for role in roles]
    while len(roles) < 8:
        roles.append(0)
    pub_pick = item['pub_pick']
    pub_win = item['pub_win']
    winrate = pub_win / pub_pick

    heroes_data.append([
        hero_id,       # id
        hero_index,    # index
        primary_attr,  # primary_attr
        attack_type,   # attack_type
        roles,         # roles
        hero_name,     # name
        pub_pick,      # pub_pick
        pub_win,       # pub_win
        winrate        # winrate
    ])
    logging.info(f'Hero id {hero_id} ({hero_name}) was added to heroes_data with index {hero_index}')

df = pd.DataFrame(heroes_data,
                  columns=[
                      'id',
                      'index',
                      'primary_attr',
                      'attack_type',
                      'roles',
                      'name',
                      'pub_pick',
                      'pub_win',
                      'winrate'])

total_picks = df["pub_pick"].sum()
df["pickrate"] = df["pub_pick"] / total_picks

if not os.path.exists('scalers'):
    os.makedirs('scalers')

winrate_scaler = MinMaxScaler(feature_range=(-1, 1))
df["winrate_scaled"] = winrate_scaler.fit_transform(df[["winrate"]])

pickrate_scaler = MinMaxScaler(feature_range=(-1, 1))
df["pickrate_scaled"] = pickrate_scaler.fit_transform(df[["pickrate"]])

df.to_json(OUTPUT_FILE, orient="records", indent=4)
logging.info(f"{OUTPUT_FILE} was saved. Heroes len = {len(df)}")
