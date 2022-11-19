from datetime import datetime, timedelta
import logging
import os

import awswrangler as wr
import boto3
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import praw
import requests
from sqlalchemy import exc, create_engine

try:
    from .schema import *
except:
    from schema import *

try:
    from .utils import *
except:
    from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[logging.FileHandler("logs/example.log"), logging.StreamHandler()],
)
logging.getLogger("requests").setLevel(logging.WARNING)  # get rid of https debug stuff
logging.info("STARTING NBA ELT PIPELINE SCRIPT Version: 1.8.5")

# helper validation function - has to be here instead of utils bc of globals().items()
def validate_schema(df: pd.DataFrame, schema: list) -> pd.DataFrame:
    """
    Schema Validation function to check whether table columns are correct before writing to SQL.
    Errors are written to Logs

    Args:
        data_df (pd.DataFrame): The Transformed Pandas DataFrame to check

        schema (list):  The corresponding columns of the Pandas DataFrame to be checked
    Returns:
        The same input DataFrame with a schema attribute that is either validated or invalidated
    """
    data_name = [k for k, v in globals().items() if v is df][0]
    try:
        if (
            len(df) == 0
        ):  # this has to be first to deal with both empty lists + valid data frames
            logging.error(f"Schema Validation Failed for {data_name}, df is empty")
            # df.schema = 'Invalidated'
            return df
        elif list(df.columns) == schema:
            logging.info(f"Schema Validation Passed for {data_name}")
            df.schema = "Validated"
            return df
        else:
            logging.error(f"Schema Validation Failed for {data_name}")
            df.schema = "Invalidated"
            return df
    except BaseException as e:
        logging.error(f"Schema Validation Failed for {data_name}, {e}")
        df.schema = "Invalidated"
        return df


logging.info("LOADED FUNCTIONS")

if __name__ == "__main__":
    logging.info("STARTING WEB SCRAPE")

    # STEP 1: Extract Raw Data
    stats = get_player_stats_data()
    boxscores = get_boxscores_data()
    injury_data = get_injuries_data()
    transactions = get_transactions_data()
    adv_stats = get_advanced_stats_data()
    odds = scrape_odds()
    reddit_data = get_reddit_data("nba")  # doesnt need transformation
    opp_stats = get_opp_stats_data()
    # schedule = schedule_scraper("2022", ["april", "may", "june"])
    shooting_stats = get_shooting_stats_data()
    twitter_tweepy_data = scrape_tweets_combo()
    reddit_comment_data = get_reddit_comments(reddit_data["reddit_url"])
    pbp_data = get_pbp_data(boxscores)  # this uses the transformed boxscores

    logging.info("FINISHED WEB SCRAPE")

    logging.info("STARTING SCHEMA VALIDATION")

    # STEP 2: Validating Schemas - 1 for each SQL Write
    stats = validate_schema(stats, stats_cols)
    adv_stats = validate_schema(adv_stats, adv_stats_cols)
    boxscores = validate_schema(boxscores, boxscores_cols)
    injury_data = validate_schema(injury_data, injury_cols)
    opp_stats = validate_schema(opp_stats, opp_stats_cols)
    pbp_data = validate_schema(pbp_data, pbp_cols)
    reddit_data = validate_schema(reddit_data, reddit_cols)
    reddit_comment_data = validate_schema(reddit_comment_data, reddit_comment_cols)
    odds = validate_schema(odds, odds_cols)
    twitter_tweepy_data = validate_schema(twitter_tweepy_data, twitter_tweepy_cols)
    transactions = validate_schema(transactions, transactions_cols)
    # schedule = validate_schema(schedule, schedule_cols)
    shooting_stats = validate_schema(shooting_stats, shooting_stats_cols)

    logging.info("FINISHED SCHEMA VALIDATION")

    logging.info("STARTING SQL STORING")

    # STEP 3: Append Transformed Data to SQL
    conn = sql_connection(os.environ.get("RDS_SCHEMA"))

    with conn.connect() as connection:
        write_to_sql_upsert(
            connection, "boxscores", boxscores, "upsert", ["player", "date"]
        )
        write_to_sql_upsert(connection, "odds", odds, "upsert", ["team", "date"])
        write_to_sql_upsert(
            connection,
            "pbp_data",
            pbp_data,
            "upsert",
            [
                "hometeam",
                "awayteam",
                "date",
                "timequarter",
                "numberperiod",
                "descriptionplayvisitor",
                "descriptionplayhome",
            ],
        )
        write_to_sql_upsert(connection, "opp_stats", opp_stats, "upsert", ["team"])
        write_to_sql_upsert(
            connection, "shooting_stats", shooting_stats, "upsert", ["player"]
        )
        write_to_sql_upsert(
            connection, "reddit_data", reddit_data, "upsert", ["reddit_url"]
        )
        write_to_sql_upsert(
            connection,
            "reddit_comment_data",
            reddit_comment_data,
            "upsert",
            ["md5_pk"],
        )
        write_to_sql_upsert(
            connection, "transactions", transactions, "upsert", ["date", "transaction"]
        )
        write_to_sql_upsert(
            connection,
            "injury_data",
            injury_data,
            "upsert",
            ["player", "team", "description"],
        )

        write_to_sql_upsert(
            connection,
            "twitter_tweepy_data",
            twitter_tweepy_data,
            "upsert",
            ["tweet_id"],
        )

        # cant upsert on these bc the column names have % and i kept getting issues
        # even after changing the col names to _pct instead etc.  no clue dude fk it
        write_to_sql(connection, "stats", stats, "append")
        write_to_sql(connection, "adv_stats", adv_stats, "append")

        # write_to_sql_upsert(connection, "schedule", schedule, "upsert", ["away_team", "home_team", "proper_date"])

    conn.dispose()

    # STEP 4: Write to S3
    write_to_s3("stats", stats)
    write_to_s3("boxscores", boxscores)
    write_to_s3("injury_data", injury_data)
    write_to_s3("transactions", transactions)
    write_to_s3("adv_stats", adv_stats)
    write_to_s3("odds", odds)
    write_to_s3("reddit_data", reddit_data)
    write_to_s3("reddit_comment_data", reddit_comment_data)
    write_to_s3("pbp_data", pbp_data)
    write_to_s3("opp_stats", opp_stats)
    write_to_s3("twitter_tweepy_data", twitter_tweepy_data)
    # write_to_s3("schedule", schedule)
    write_to_s3("shooting_stats", shooting_stats)

    # STEP 5: Grab Logs from previous steps & send email out detailing notable events
    logs = pd.read_csv("logs/example.log", sep=r"\\t", engine="python", header=None)
    logs = logs.rename(columns={0: "errors"})
    logs = logs.query("errors.str.contains('Failed')", engine="python")

    # STEP 6: Send Email
    send_aws_email(logs)
    logging.info("FINISHED NBA ELT PIPELINE SCRIPT Version: 1.8.5")
