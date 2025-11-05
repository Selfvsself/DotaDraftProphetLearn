import requests
import csv
import time
import logging

LAST_MATCH_ID = 8473111211
BASE_URL = "https://api.opendota.com/api/publicMatches?less_than_match_id={match_id}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

COUNTER = 1
ATTEMPT = 0
MAX_ATTEMPT = 10
OUTPUT_FILENAME = f"../data/raw/matches_{LAST_MATCH_ID}.csv"

fields = [
    "match_id",
    "radiant_win",
    "duration",
    "lobby_type",
    "game_mode",
    "avg_rank_tier",
    "num_rank_tier",
    "cluster",
    "radiant_team",
    "dire_team"
]

with open(OUTPUT_FILENAME, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    while True:
        url = BASE_URL.format(match_id=LAST_MATCH_ID)
        logging.info(f"GET {url}")

        try:
            response = requests.get(url, timeout=20)
        except requests.RequestException as e:
            logging.info(f"Exception: {e}, attempt №{ATTEMPT + 1}")
            if ATTEMPT >= MAX_ATTEMPT:
                logging.info("Retries exceeded, exiting.")
                break
            ATTEMPT += 1
            continue
        ATTEMPT = 0

        if response.status_code != 200:
            logging.info(f"Response code {response.status_code}, exiting.")
            break

        data = response.json()
        if not data:
            logging.info("Empty response, exiting.")
            break

        for item in data:
            row = {field: str(item.get(field, "")) for field in fields}
            writer.writerow(row)

        LAST_MATCH_ID = data[-1].get("match_id")
        logging.info(f"Processed {len(data) * COUNTER} records, next match_id = {LAST_MATCH_ID}")

        time.sleep(10)
        COUNTER += 1

logging.info(f"Records saved to {OUTPUT_FILENAME}")
