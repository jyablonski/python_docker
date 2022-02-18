import os
import logging
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import praw
from bs4 import BeautifulSoup
from sqlalchemy import exc, create_engine
import boto3
from botocore.exceptions import ClientError
from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[logging.FileHandler("logs/example.log"), logging.StreamHandler()],
)
logging.getLogger("requests").setLevel(logging.WARNING)  # get rid of https debug stuff

logging.info("STARTING NBA ELT PIPELINE SCRIPT Version: 1.2.8")
# logging.warning("STARTING NBA ELT PIPELINE SCRIPT Version: 1.2.8")
# logging.error("STARTING NBA ELT PIPELINE SCRIPT Version: 1.2.8")

# helper sql function - has to be here & not utils bc of globals().items()
def write_to_sql(con, data, table_type):
    """
    SQL Table function to write a pandas data frame in aws_dfname_source format

    Args:
        data: The Pandas DataFrame to store in SQL

        table_type: Whether the table should replace or append to an existing SQL Table under that name

    Returns:
        Writes the Pandas DataFrame to a Table in Snowflake in the {nba_source} Schema we connected to.

    """
    try:
        data_name = [k for k, v in globals().items() if v is data][0]
        # ^ this disgusting monstrosity is to get the name of the -fucking- dataframe lmfao
        if len(data) == 0:
            logging.info(f"{data_name} is empty, not writing to SQL")
        else:
            data.to_sql(
                con=con,
                name=f"aws_{data_name}_source",
                index=False,
                if_exists=table_type,
            )
            logging.info(f"Writing aws_{data_name}_source to SQL")
    except BaseException as error:
        logging.error(f"SQL Write Script Failed, {error}")
        return error


# helper validation function - has to be here instead of utils bc of globals().items()
def validate_schema(data_df, schema):
    """
    Schema Validation function to check whether table columns are correct before writing to SQL.
    Errors are written to Logs

    Args:
        data_df (pd.DataFrame): The Transformed Pandas DataFrame to check

        schema (list):  The corresponding columns of the Pandas DataFrame to be checked
    Returns:
        None
    """
    data_name = [k for k, v in globals().items() if v is data_df][0]
    try:
        if (
            len(data_df) == 0
        ):  # this has to be first to deal with both empty lists + valid data frames
            logging.error(f"Schema Validation Failed for {data_name}, df is empty")
        elif list(data_df.columns) == schema:
            logging.info(f"Schema Validation Passed for {data_name}")
        else:
            logging.error(f"Schema Validation Failed for {data_name}")
    except BaseException as e:
        logging.error(f"Schema Validation Failed for {data_name}, {e}")
        pass


logging.info("Starting Logging Function")

logging.info("LOADED FUNCTIONS")

today = datetime.now().date()
todaytime = datetime.now()
yesterday = today - timedelta(1)
day = (datetime.now() - timedelta(1)).day
month = (datetime.now() - timedelta(1)).month
year = (datetime.now() - timedelta(1)).year
if datetime.now().date() < datetime(2022, 4, 11).date():
    season_type = "Regular Season"
else:
    season_type = "Playoffs"


if __name__ == "__main__":
    logging.info("STARTING WEB SCRAPE")

    # STEP 1: Extract Raw Data
    stats_raw = get_player_stats_data()
    boxscores_raw = get_boxscores_data()
    injury_data_raw = get_injuries_data()
    transactions_raw = get_transactions_data()
    adv_stats_raw = get_advanced_stats_data()
    odds_raw = get_odds_data()
    reddit_data = get_reddit_data("nba")  # doesnt need transformation
    opp_stats_raw = get_opp_stats_data()
    twitter_data = scrape_tweets("nba")

    logging.info("FINISHED WEB SCRAPE")

    # STEP 2: Transform data
    logging.info("STARTING DATA TRANSFORMATIONS")

    stats = get_player_stats_transformed(stats_raw)
    boxscores = get_boxscores_transformed(boxscores_raw)
    injury_data = get_injuries_transformed(injury_data_raw)
    transactions = get_transactions_transformed(transactions_raw)
    adv_stats = get_advanced_stats_transformed(adv_stats_raw)
    odds = get_odds_transformed(odds_raw)
    reddit_comment_data = get_reddit_comments(reddit_data["reddit_url"])
    pbp_data = get_pbp_data_transformed(
        boxscores
    )  # this uses the transformed boxscores
    opp_stats = get_opp_stats_transformed(opp_stats_raw)

    logging.info("FINISHED DATA TRANSFORMATIONS")

    logging.info("STARTING SCHEMA VALIDATION")

    # STEP 3: Validating Schemas - 1 for each SQL Write
    validate_schema(stats, stats_cols)
    validate_schema(adv_stats, adv_stats_cols)
    validate_schema(boxscores, boxscores_cols)
    validate_schema(injury_data, injury_cols)
    validate_schema(opp_stats, opp_stats_cols)
    validate_schema(pbp_data, pbp_cols)
    validate_schema(reddit_data, reddit_cols)
    validate_schema(reddit_comment_data, reddit_comment_cols)
    validate_schema(odds, odds_cols)
    validate_schema(transactions, transactions_cols)
    validate_schema(twitter_data, twitter_cols)

    logging.info("FINISHED SCHEMA VALIDATION")

    logging.info("STARTING SQL STORING")

    # STEP 4: Append Transformed Data to SQL
    conn = sql_connection(os.environ.get("RDS_SCHEMA"))
    write_to_sql(conn, stats, "append")
    write_to_sql(conn, boxscores, "append")
    write_to_sql(conn, injury_data, "append")
    write_to_sql(conn, transactions, "append")
    write_to_sql(conn, adv_stats, "append")
    write_to_sql(conn, odds, "append")
    write_to_sql(conn, reddit_data, "append")
    write_to_sql(conn, reddit_comment_data, "append")
    write_to_sql(conn, pbp_data, "append")
    write_to_sql(conn, opp_stats, "append")
    write_to_sql(conn, twitter_data, "append")

    # STEP 5: Grab Logs from previous steps & send email out detailing notable events
    logs = pd.read_csv("logs/example.log", sep=r"\\t", engine="python", header=None)
    logs = logs.rename(columns={0: "errors"})
    logs = logs.query("errors.str.contains('Failed')", engine="python")
    execute_email_function(logs)

logging.info("FINISHED NBA ELT PIPELINE SCRIPT Version: 1.2.8")
