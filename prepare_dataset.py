import ast
import csv
import json
from os import listdir

from tqdm import tqdm

hero_id_to_index = {}
with open("docs/heroes_dict.json", "r") as file:
    full_data = json.load(file)
    for item in full_data:
        hero_id_to_index[item['key']] = item['value']


def parse_line_to_list(line: str) -> list[int]:
    return list(map(int, ast.literal_eval(line)))


def map_team(team_name: str, ids: list[int]) -> list[int]:
    result = []
    for hero_id in ids:
        if hero_id not in hero_id_to_index:
            raise ValueError(f"Unknown hero id in {team_name}: {hero_id}")
        result.append(hero_id_to_index[hero_id])
    return result


def validate_and_map_heroes(
        radiant_array: list[int],
        dire_array: list[int]
) -> tuple[list[int], list[int]]:
    if len(radiant_array) != 5:
        raise ValueError(f"Radiant team must have 5 heroes, got {len(radiant_array)}")
    if len(dire_array) != 5:
        raise ValueError(f"Dire team must have 5 heroes, got {len(dire_array)}")

    combined = radiant_array + dire_array
    if len(set(combined)) != len(combined):
        raise ValueError("Radiant and Dire teams must have different heroes")

    radiant = map_team("Radiant", radiant_array)
    dire = map_team("Dire", dire_array)

    return radiant, dire


def normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0

    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0, min(1, normalized))
    return round(normalized, 2)


def normalize_avg_rank_tier(rank_tier: float) -> float:
    return normalize(rank_tier, 11, 75)


def normalize_num_rank_tier(rank_tier: float) -> float:
    return normalize(rank_tier, 1, 10)


def normalize_duration(duration: float) -> float:
    return normalize(duration, 362, 8176)


dir_path = 'data/'
allow_game_types = ['1', '2', '16', '22']

output_file = 'datasets/dataset.csv'

count = 0
r_win_count = 0
d_win_count = 0
with open(output_file, mode='w', newline='') as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(
        ['match_id', 'result', 'avg_rank_tier', 'num_rank_tier', 'duration', 'radiant_vector', 'dire_vector'])

    for file in tqdm(listdir(dir_path)):
        with open(dir_path + file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                game_type = row['game_mode']
                match_id = row['match_id']
                radiant_win = row['radiant_win']
                avg_rank_tier = int(row['avg_rank_tier'])
                num_rank_tier = int(row['num_rank_tier'])
                duration = int(row['duration'])

                if game_type in allow_game_types:
                    if radiant_win == 'True':
                        result = 1
                        r_win_count += 1
                    else:
                        result = 0
                        d_win_count += 1

                    try:
                        radiant_array = parse_line_to_list(row['radiant_team'])
                        dire_array = parse_line_to_list(row['dire_team'])

                        vector = validate_and_map_heroes(radiant_array, dire_array)
                        writer.writerow([
                            match_id,
                            result,
                            normalize_avg_rank_tier(avg_rank_tier),
                            normalize_num_rank_tier(num_rank_tier),
                            normalize_duration(duration),
                            json.dumps(vector[0]),  # сохраняем как JSON-строку
                            json.dumps(vector[1])
                        ])
                    except Exception as e:
                        print(row, e)
                        continue
                    count += 1
print(f'Всего записей: {count}')
print(f'Побед: {r_win_count}')
print(f'Поражений: {d_win_count}')
