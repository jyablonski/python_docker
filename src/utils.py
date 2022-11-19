from datetime import datetime, timedelta
import hashlib
import logging
import os
import requests
from typing import List, Optional
import uuid

import awswrangler as wr
import boto3
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import praw
import requests
from sqlalchemy import exc, create_engine
from sqlalchemy.engine.base import Engine
import sentry_sdk
import tweepy
from tweepy import OAuthHandler

sentry_sdk.init(os.environ.get("SENTRY_TOKEN"), traces_sample_rate=1.0)
sentry_sdk.set_user({"email": "jyablonski9@gmail.com"})


def get_season_type(todays_date: datetime.date = datetime.now().date()) -> str:
    """
    Function to generate Season Type for a given Date.  Defaults to today's date.

    Args:
        todays_date (date): The Date to generate a Season Type for

    Returns:
        The Season Type for Given Date
    """
    if todays_date < datetime(2023, 4, 9).date():
        season_type = "Regular Season"
    elif (todays_date >= datetime(2023, 4, 9).date()) & (
        todays_date < datetime(2023, 4, 16).date()
    ):
        season_type = "Play-In"
    else:
        season_type = "Playoffs"

    return season_type


def add_sentiment_analysis(df: pd.DataFrame, sentiment_col: str) -> pd.DataFrame:
    """
    Function to add Sentiment Analysis columns via nltk Vader Lexicon.

    Args:
        df (pd.DataFrame): the Pandas DataFrame

        sentiment_col (str): The Column in the DataFrame to run Sentiment Analysis on (comments / tweets etc).

    Returns:
        The same DataFrame but with the Sentiment Analysis columns attached.
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        df["compound"] = [
            analyzer.polarity_scores(x)["compound"] for x in df[sentiment_col]
        ]
        df["neg"] = [analyzer.polarity_scores(x)["neg"] for x in df[sentiment_col]]
        df["neu"] = [analyzer.polarity_scores(x)["neu"] for x in df[sentiment_col]]
        df["pos"] = [analyzer.polarity_scores(x)["pos"] for x in df[sentiment_col]]
        df["sentiment"] = np.where(df["compound"] > 0, 1, 0)
        return df
    except BaseException as e:
        logging.error(f"Error Occurred while adding Sentiment Analysis, {e}")
        sentry_sdk.capture_exception(e)
        raise e


def get_leading_zeroes(month: int) -> str:
    """
    Function to add leading zeroes to a month (1 (January) -> 01) for the write_to_s3 function.

    Args:
        month (int): The month integer

    Returns:
        The same month integer with a leading 0 if it is less than 10 (Nov/Dec aka 11/12 unaffected).
    """
    if len(str(month)) > 1:
        return month
    else:
        return f"0{month}"


def clean_player_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to remove suffixes from player names for joining downstream.
    Assumes the column name is ['player']

    Args:
        df (DataFrame): The DataFrame you wish to alter

    Returns:
        df with transformed player names
    """
    try:
        df["player"] = df["player"].str.replace(" Jr.", "", regex=True)
        df["player"] = df["player"].str.replace(" Sr.", "", regex=True)
        df["player"] = df["player"].str.replace(
            " III", "", regex=True
        )  # III HAS TO GO FIRST, OVER II
        df["player"] = df["player"].str.replace(
            " II", "", regex=True
        )  # Robert Williams III -> Robert WilliamsI
        df["player"] = df["player"].str.replace(" IV", "", regex=True)
        return df
    except BaseException as e:
        logging.error(f"Error Occurred with clean_player_names, {e}")
        sentry_sdk.capture_exception(e)


def get_player_stats_data() -> pd.DataFrame:
    """
    Web Scrape function w/ BS4 that grabs aggregate season stats

    Args:
        None

    Returns:
        DataFrame of Player Aggregate Season stats
    """
    # stats = stats.rename(columns={"fg%": "fg_pct", "3p%": "3p_pct", "2p%": "2p_pct", "efg%": "efg_pct", "ft%": "ft_pct"})
    try:
        year_stats = 2023
        url = f"https://www.basketball-reference.com/leagues/NBA_{year_stats}_per_game.html"
        html = requests.get(url).content
        soup = BeautifulSoup(html, "html.parser")
        headers = [th.getText() for th in soup.findAll("tr", limit=2)[0].findAll("th")]
        headers = headers[1:]
        rows = soup.findAll("tr")[1:]
        player_stats = [
            [td.getText() for td in rows[i].findAll("td")] for i in range(len(rows))
        ]
        stats = pd.DataFrame(player_stats, columns=headers)
        stats["PTS"] = pd.to_numeric(stats["PTS"])
        stats = stats.query("Player == Player").reset_index()
        stats["Player"] = (
            stats["Player"]
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )
        stats.columns = stats.columns.str.lower()
        stats["scrape_date"] = datetime.now().date()
        stats = stats.drop("index", axis=1)
        logging.info(
            f"General Stats Transformation Function Successful, retrieving {len(stats)} updated rows"
        )
        return stats
    except BaseException as error:
        logging.error(f"General Stats Extraction Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        df = []
        return df


def get_boxscores_data(
    month: int = (datetime.now() - timedelta(1)).month,
    day: int = (datetime.now() - timedelta(1)).day,
    year: int = (datetime.now() - timedelta(1)).year,
) -> pd.DataFrame:
    """
    Function that grabs box scores from a given date in mmddyyyy format - defaults to yesterday.  values can be ex. 1 or 01.
    Can't use read_html for this so this is raw web scraping baby.

    Args:
        month (int): month value of the game played (0 - 12)

        day (int): day value of the game played (1 - 31)

        year (int): year value of the game played (2021)

    Returns:
        DataFrame of Player Aggregate Season stats
    """
    url = f"https://www.basketball-reference.com/friv/dailyleaders.fcgi?month={month}&day={day}&year={year}&type=all"
    season_type = get_season_type()

    try:
        html = requests.get(url).content
        soup = BeautifulSoup(html, "html.parser")
        headers = [th.getText() for th in soup.findAll("tr", limit=2)[0].findAll("th")]
        headers = headers[1:]
        headers[1] = "Team"
        headers[2] = "Location"
        headers[3] = "Opponent"
        headers[4] = "Outcome"
        headers[6] = "FGM"
        headers[8] = "FGPercent"
        headers[9] = "threePFGMade"
        headers[10] = "threePAttempted"
        headers[11] = "threePointPercent"
        headers[14] = "FTPercent"
        headers[15] = "OREB"
        headers[16] = "DREB"
        headers[24] = "PlusMinus"

        rows = soup.findAll("tr")[1:]
        player_stats = [
            [td.getText() for td in rows[i].findAll("td")] for i in range(len(rows))
        ]

        df = pd.DataFrame(player_stats, columns=headers)

        df[
            [
                "FGM",
                "FGA",
                "FGPercent",
                "threePFGMade",
                "threePAttempted",
                "threePointPercent",
                "OREB",
                "DREB",
                "TRB",
                "AST",
                "STL",
                "BLK",
                "TOV",
                "PF",
                "PTS",
                "PlusMinus",
                "GmSc",
            ]
        ] = df[
            [
                "FGM",
                "FGA",
                "FGPercent",
                "threePFGMade",
                "threePAttempted",
                "threePointPercent",
                "OREB",
                "DREB",
                "TRB",
                "AST",
                "STL",
                "BLK",
                "TOV",
                "PF",
                "PTS",
                "PlusMinus",
                "GmSc",
            ]
        ].apply(
            pd.to_numeric
        )
        df["date"] = str(year) + "-" + str(month) + "-" + str(day)
        df["date"] = pd.to_datetime(df["date"])
        df["Type"] = season_type
        df["Season"] = 2022
        df["Location"] = df["Location"].apply(lambda x: "A" if x == "@" else "H")
        df["Team"] = df["Team"].str.replace("PHO", "PHX")
        df["Team"] = df["Team"].str.replace("CHO", "CHA")
        df["Team"] = df["Team"].str.replace("BRK", "BKN")
        df["Opponent"] = df["Opponent"].str.replace("PHO", "PHX")
        df["Opponent"] = df["Opponent"].str.replace("CHO", "CHA")
        df["Opponent"] = df["Opponent"].str.replace("BRK", "BKN")
        df = df.query("Player == Player").reset_index(drop=True)
        df["Player"] = (
            df["Player"]
            .str.normalize("NFKD")  # this is removing all accented characters
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )
        df["scrape_date"] = datetime.now().date()
        df.columns = df.columns.str.lower()
        logging.info(
            f"Box Score Transformation Function Successful, retrieving {len(df)} rows for {year}-{month}-{day}"
        )
        return df
    except IndexError as error:
        logging.warning(
            f"Box Score Extraction Function Failed, {error}, no data available for {year}-{month}-{day}"
        )
        sentry_sdk.capture_exception(error)
        df = []
        return df
    except BaseException as error:
        logging.error(f"Box Score Extraction Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        df = []
        return df


def get_opp_stats_data() -> pd.DataFrame:
    """
    Web Scrape function w/ pandas read_html that grabs all regular season opponent team stats

    Args:
        None

    Returns:
        Pandas DataFrame of all current team opponent stats
    """
    year = (datetime.now() - timedelta(1)).year
    month = (datetime.now() - timedelta(1)).month
    day = (datetime.now() - timedelta(1)).day
    year_stats = 2023

    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year_stats}.html"
        df = pd.read_html(url)[5]
        df = df[["Team", "FG%", "3P%", "3P", "PTS"]]
        df = df.rename(
            columns={
                df.columns[0]: "team",
                df.columns[1]: "fg_percent_opp",
                df.columns[2]: "threep_percent_opp",
                df.columns[3]: "threep_made_opp",
                df.columns[4]: "ppg_opp",
            }
        )
        df = df.query('team != "League Average"')
        df = df.reset_index(drop=True)
        df["scrape_date"] = datetime.now().date()
        logging.info(
            f"Opp Stats Transformation Function Successful, retrieving {len(df)} rows for {year}-{month}-{day}"
        )
        return df
    except BaseException as error:
        logging.error(f"Opp Stats Web Scrape Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        df = []
        return df


def get_injuries_data() -> pd.DataFrame:
    """
    Web Scrape function w/ pandas read_html that grabs all current injuries

    Args:
        None

    Returns:
        Pandas DataFrame of all current player injuries & their associated team
    """
    try:
        url = "https://www.basketball-reference.com/friv/injuries.fcgi"
        df = pd.read_html(url)[0]
        df = df.rename(columns={"Update": "Date"})
        df.columns = df.columns.str.lower()
        df["scrape_date"] = datetime.now().date()
        df["player"] = (
            df["player"]
            .str.normalize("NFKD")  # this is removing all accented characters
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )
        df = clean_player_names(df)
        logging.info(
            f"Injury Transformation Function Successful, retrieving {len(df)} rows"
        )
        return df
    except BaseException as error:
        logging.error(f"Injury Web Scrape Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        df = []
        return df


def get_transactions_data() -> pd.DataFrame:
    """
    Web Scrape function w/ BS4 that retrieves NBA Trades, signings, waivers etc.

    Args:
        None

    Returns:
        Pandas DataFrame of all season transactions, trades, player waives etc.
    """
    try:
        url = "https://www.basketball-reference.com/leagues/NBA_2023_transactions.html"
        html = requests.get(url).content
        soup = BeautifulSoup(html, "html.parser")
        # theres a bunch of garbage in the first 50 rows - no matter what
        trs = soup.findAll("li")[70:]
        rows = []
        mylist = []
        for tr in trs:
            date = tr.find("span")
            # needed bc span can be null (multi <p> elements per span)
            if date is not None:
                date = date.text
            data = tr.findAll("p")
            for p in data:
                mylist.append(p.text)
            data3 = [date] + [mylist]
            rows.append(data3)
            mylist = []

        transactions = pd.DataFrame(rows)
        transactions.columns = ["Date", "Transaction"]
        transactions = transactions.query(
            'Date == Date & Date != ""'
        ).reset_index()  # filters out nulls and empty values
        transactions = transactions.explode("Transaction")
        transactions["Date"] = transactions["Date"].str.replace(
            "?", "Jan 1, 2021", regex=True  # bad data 10-14-21
        )
        transactions["Date"] = pd.to_datetime(transactions["Date"])
        transactions.columns = transactions.columns.str.lower()
        transactions = transactions[["date", "transaction"]]
        transactions["scrape_date"] = datetime.now().date()
        logging.info(
            f"Transactions Transformation Function Successful, retrieving {len(transactions)} rows"
        )
        return transactions
    except BaseException as error:
        logging.error(f"Transaction Web Scrape Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        df = []
        return df


def get_advanced_stats_data() -> pd.DataFrame:
    """
    Web Scrape function w/ pandas read_html that grabs all team advanced stats

    Args:
        None

    Returns:
        DataFrame of all current Team Advanced Stats
    """
    year_stats = 2023
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year_stats}.html"
        df = pd.read_html(url)
        df = pd.DataFrame(df[10])
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        df.columns = [
            "Team",
            "Age",
            "W",
            "L",
            "PW",
            "PL",
            "MOV",
            "SOS",
            "SRS",
            "ORTG",
            "DRTG",
            "NRTG",
            "Pace",
            "FTr",
            "3PAr",
            "TS%",
            "bby1",  # the bby columns are because of hierarchical html formatting - they're just blank columns
            "eFG%",
            "TOV%",
            "ORB%",
            "FT/FGA",
            "bby2",
            "eFG%_opp",
            "TOV%_opp",
            "DRB%_opp",
            "FT/FGA_opp",
            "bby3",
            "Arena",
            "Attendance",
            "Att/Game",
        ]
        df.drop(["bby1", "bby2", "bby3"], axis=1, inplace=True)
        df = df.query('Team != "League Average"').reset_index()
        # Playoff teams get a * next to them ??  fkn stupid, filter it out.
        df["Team"] = df["Team"].str.replace("*", "", regex=True)
        df["scrape_date"] = datetime.now().date()
        df.columns = df.columns.str.lower()
        logging.info(
            f"Advanced Stats Transformation Function Successful, retrieving updated data for 30 Teams"
        )
        return df
    except BaseException as error:
        logging.error(f"Advanced Stats Web Scrape Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        df = []
        return df


def get_shooting_stats_data() -> pd.DataFrame:
    """
    Web Scrape function w/ pandas read_html that grabs all raw shooting stats

    Args:
        None

    Returns:
        DataFrame of raw shooting stats
    """
    year_stats = 2023
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year_stats}_shooting.html"
        df = pd.read_html(url)[0]
        df.columns = df.columns.to_flat_index()
        df = df.rename(
            columns={
                df.columns[1]: "player",
                df.columns[6]: "mp",
                df.columns[8]: "avg_shot_distance",
                df.columns[10]: "pct_fga_2p",
                df.columns[11]: "pct_fga_0_3",
                df.columns[12]: "pct_fga_3_10",
                df.columns[13]: "pct_fga_10_16",
                df.columns[14]: "pct_fga_16_3p",
                df.columns[15]: "pct_fga_3p",
                df.columns[18]: "fg_pct_0_3",
                df.columns[19]: "fg_pct_3_10",
                df.columns[20]: "fg_pct_10_16",
                df.columns[21]: "fg_pct_16_3p",
                df.columns[24]: "pct_2pfg_ast",
                df.columns[25]: "pct_3pfg_ast",
                df.columns[27]: "dunk_pct_tot_fg",
                df.columns[28]: "dunks",
                df.columns[30]: "corner_3_ast_pct",
                df.columns[31]: "corner_3pm_pct",
                df.columns[33]: "heaves_att",
                df.columns[34]: "heaves_makes",
            }
        )[
            [
                "player",
                "mp",
                "avg_shot_distance",
                "pct_fga_2p",
                "pct_fga_0_3",
                "pct_fga_3_10",
                "pct_fga_10_16",
                "pct_fga_16_3p",
                "pct_fga_3p",
                "fg_pct_0_3",
                "fg_pct_3_10",
                "fg_pct_10_16",
                "fg_pct_16_3p",
                "pct_2pfg_ast",
                "pct_3pfg_ast",
                "dunk_pct_tot_fg",
                "dunks",
                "corner_3_ast_pct",
                "corner_3pm_pct",
                "heaves_att",
                "heaves_makes",
            ]
        ]
        df = df.query('player != "Player"').copy()
        df["mp"] = pd.to_numeric(df["mp"])
        df = (
            df.sort_values(["mp"], ascending=False)
            .groupby("player")
            .first()
            .reset_index()
            .drop("mp", axis=1)
        )
        df["player"] = (
            df["player"]
            .str.normalize("NFKD")  # this is removing all accented characters
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )
        df = clean_player_names(df)
        df["scrape_date"] = datetime.now().date()
        df["scrape_ts"] = datetime.now()
        logging.info(
            f"Shooting Stats Transformation Function Successful, retrieving {len(df)} rows"
        )
        return df
    except BaseException as error:
        logging.error(f"Shooting Stats Web Scrape Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        df = []
        return df


def scrape_odds():
    """
    Function to web scrape Gambling Odds from cover.com

    Args:
        None

    Returns:
        DataFrame of Gambling Odds for Today's Games
    """
    try:
        url = "https://www.covers.com/sport/basketball/nba/odds"
        df = pd.read_html(url)
        odds = df[0]
        odds["spread"] = df[3]["Unnamed: 1"]
        odds["moneyline"] = df[1]["Unnamed: 1"]
        odds = odds[["Time (ET)", "Game (ET)", "spread", "moneyline"]]
        odds = odds.rename(columns={"Time (ET)": "datetime1", "Game (ET)": "team"})
        odds = odds.query("datetime1.str.contains('Today')", engine="python").copy()
        start_times = odds["datetime1"]
        odds["spread"] = odds["spread"].str.replace("PK", "-1.0")
        odds["spread"] = odds["spread"].str.replace("\+ ", "", regex=True)
        odds["spread"] = odds["spread"].str.replace(" \+ ", "", regex=True)
        odds["spread"] = odds["spread"].str.replace(" \+", "", regex=True)
        odds["spread"] = odds["spread"].str.replace("95", "")
        # + is a special character, have to escape it with \ and set regex = true to avoiod an error
        odds["spread"] = odds["spread"].str.replace("\+95", "", regex=True)
        odds["spread"] = odds["spread"].str.replace("100", "")
        odds["spread"] = odds["spread"].str.replace("\+100", "", regex=True)
        odds["spread"] = odds["spread"].str.replace("-105", "")
        odds["spread"] = odds["spread"].str.replace("-110", "")
        odds["spread"] = odds["spread"].str.replace("-115", "")
        odds["spread"] = odds["spread"].str.replace("-120", "")
        odds["spread"] = odds["spread"].str.replace("-125", "")
        odds["datetime1"] = odds["datetime1"].str.replace("Today, ", "")
        odds_split = odds[["datetime1", "team", "spread", "moneyline"]]
        odds_final = odds_split.copy()
        # turning the space separated elements in a list, then exploding that list
        odds_final["team"] = odds_final["team"].str.split(" ", n=1, expand=False)
        odds_final["spread"] = odds_final["spread"].str.split(" ", n=1, expand=False)
        odds_final["moneyline"] = odds_final["moneyline"].str.split(
            " ", n=1, expand=False
        )
        # # odds_final.set_index(['teams'])
        odds_final = odds_final.explode(["team", "spread", "moneyline"]).reset_index()
        odds_final = odds_final.drop("index", axis=1)
        odds_final["date"] = datetime.now().date()
        odds_final["spread"] = odds_final[
            "spread"
        ].str.strip()  # strip trailing and leading spaces
        odds_final["moneyline"] = odds_final["moneyline"].str.strip()
        odds_final["datetime1"] = pd.to_datetime(
            str(datetime.now().date()) + " " + odds_final["datetime1"]
        )
        odds_final["total"] = 200
        odds_final["team"] = odds_final["team"].str.replace("BK", "BKN")
        odds_final["moneyline"] = odds_final["moneyline"].str.replace(
            "+", "", regex=True
        )
        odds_final["moneyline"] = odds_final["moneyline"].astype("int")
        odds_final = odds_final[
            ["team", "spread", "total", "moneyline", "date", "datetime1"]
        ]

        logging.info(
            f"Odds Scrape Successful, returning {len(odds_final)} records from {len(odds_final) / 2} games Today"
        )
        return odds_final
    except BaseException as e:
        logging.error(f"Odds Function Web Scrape Failed, {e}")
        sentry_sdk.capture_exception(e)
        df = []
        return df


def get_odds_data() -> pd.DataFrame:
    """
    *********** DEPRECATED AS OF 2022-10-19 ***********

    Web Scrape function w/ pandas read_html that grabs current day's nba odds in raw format.
    There are 2 objects [0], [1] if the days are split into 2.
    AWS ECS operates in UTC time so the game start times are actually 5-6+ hours ahead of what they actually are, so there are 2 html tables.

    Args:
        None

    Returns:
        Pandas DataFrame of NBA moneyline + spread odds for upcoming games for that day
    """
    year = (datetime.now() - timedelta(1)).year

    try:
        url = "https://sportsbook.draftkings.com/leagues/basketball/nba"
        df = pd.read_html(url)
        if len(df) == 0:
            logging.info(f"Odds Transformation Failed, no Odds Data available.")
            df = []
            return df
        else:
            try:
                data1 = df[0].copy()
                data1.columns.values[0] = "Tomorrow"
                date_try = str(year) + " " + data1.columns[0]
                data1["date"] = np.where(
                    date_try == "2022 Tomorrow",
                    datetime.now().date(),  # if the above is true, then return this
                    str(year) + " " + data1.columns[0],  # if false then return this
                )
                # )
                date_try = data1["date"].iloc[0]
                data1.reset_index(drop=True)
                data1["Tomorrow"] = data1["Tomorrow"].str.replace(
                    "LA Clippers", "LAC Clippers", regex=True
                )

                data1["Tomorrow"] = data1["Tomorrow"].str.replace(
                    "AM", "AM ", regex=True
                )
                data1["Tomorrow"] = data1["Tomorrow"].str.replace(
                    "PM", "PM ", regex=True
                )
                data1["Time"] = data1["Tomorrow"].str.split().str[0]
                data1["datetime1"] = (
                    pd.to_datetime(date_try.strftime("%Y-%m-%d") + " " + data1["Time"])
                    - timedelta(hours=6)
                    + timedelta(days=1)
                )
                if len(df) > 1:  # if more than 1 day's data appears then do this
                    data2 = df[1].copy()
                    data2.columns.values[0] = "Tomorrow"
                    data2.reset_index(drop=True)
                    data2["Tomorrow"] = data2["Tomorrow"].str.replace(
                        "LA Clippers", "LAC Clippers", regex=True
                    )
                    data2["Tomorrow"] = data2["Tomorrow"].str.replace(
                        "AM", "AM ", regex=True
                    )
                    data2["Tomorrow"] = data2["Tomorrow"].str.replace(
                        "PM", "PM ", regex=True
                    )
                    data2["Time"] = data2["Tomorrow"].str.split().str[0]
                    data2["datetime1"] = (
                        pd.to_datetime(
                            date_try.strftime("%Y-%m-%d") + " " + data2["Time"]
                        )
                        - timedelta(hours=6)
                        + timedelta(days=1)
                    )
                    data2["date"] = data2["datetime1"].dt.date

                    data = data1.append(data2).reset_index(drop=True)
                    data["SPREAD"] = data["SPREAD"].str[:-4]
                    data["TOTAL"] = data["TOTAL"].str[:-4]
                    data["TOTAL"] = data["TOTAL"].str[2:]
                    data["Tomorrow"] = data["Tomorrow"].str.split().str[1:2]
                    data["Tomorrow"] = pd.DataFrame(
                        [
                            str(line).strip("[").strip("]").replace("'", "")
                            for line in data["Tomorrow"]
                        ]
                    )
                    data["SPREAD"] = data["SPREAD"].str.replace("pk", "-1", regex=True)
                    data["SPREAD"] = data["SPREAD"].str.replace("+", "", regex=True)
                    data.columns = data.columns.str.lower()
                    data = data[
                        [
                            "tomorrow",
                            "spread",
                            "total",
                            "moneyline",
                            "date",
                            "datetime1",
                        ]
                    ]
                    data = data.rename(columns={data.columns[0]: "team"})
                    data = data.query(
                        "date == date.min()"
                    )  # only grab games from upcoming day
                    logging.info(
                        f"Odds Transformation Function Successful {len(df)} day, retrieving {len(data)} rows"
                    )
                    return data
                else:  # if there's only 1 day of data then just use that
                    data = data1.reset_index(drop=True)
                    data["SPREAD"] = data["SPREAD"].str[:-4]
                    data["TOTAL"] = data["TOTAL"].str[:-4]
                    data["TOTAL"] = data["TOTAL"].str[2:]
                    data["Tomorrow"] = data["Tomorrow"].str.split().str[1:2]
                    data["Tomorrow"] = pd.DataFrame(
                        [
                            str(line).strip("[").strip("]").replace("'", "")
                            for line in data["Tomorrow"]
                        ]
                    )
                    data["SPREAD"] = data["SPREAD"].str.replace("pk", "-1", regex=True)
                    data["SPREAD"] = data["SPREAD"].str.replace("+", "", regex=True)
                    data.columns = data.columns.str.lower()
                    data = data[
                        [
                            "tomorrow",
                            "spread",
                            "total",
                            "moneyline",
                            "date",
                            "datetime1",
                        ]
                    ]
                    data = data.rename(columns={data.columns[0]: "team"})
                    data = data.query(
                        "date == date.min()"
                    )  # only grab games from upcoming day
                    logging.info(
                        f"Odds Transformation Successful {len(df)} day, retrieving {len(data)} rows"
                    )
                    return data
            except BaseException as error:
                logging.error(
                    f"Odds Transformation Failed for {len(df)} day objects, {error}"
                )
                sentry_sdk.capture_exception(error)
                data = []
                return data
    except (
        BaseException,
        ValueError,
    ) as error:  # valueerror fucked shit up apparently idfk
        logging.error(f"Odds Function Web Scrape Failed, {error}")
        sentry_sdk.capture_exception(error)
        df = []
        return df


def get_reddit_data(sub: str = "nba") -> pd.DataFrame:
    """
    Web Scrape function w/ PRAW that grabs top ~27 top posts from a given subreddit.
    Left sub as an argument in case I want to scrape multi subreddits in the future (r/nba, r/nbadiscussion, r/sportsbook etc)

    Args:
        sub (string): subreddit to query

    Returns:
        Pandas DataFrame of all current top posts on r/nba
    """
    reddit = praw.Reddit(
        client_id=os.environ.get("reddit_accesskey"),
        client_secret=os.environ.get("reddit_secretkey"),
        user_agent="praw-app",
        username=os.environ.get("reddit_user"),
        password=os.environ.get("reddit_pw"),
    )
    try:
        subreddit = reddit.subreddit(sub)
        posts = []
        for post in subreddit.hot(limit=27):
            posts.append(
                [
                    post.title,
                    post.score,
                    post.id,
                    post.url,
                    str(f"https://www.reddit.com{post.permalink}"),
                    post.num_comments,
                    post.selftext,
                    datetime.now().date(),
                    datetime.now(),
                ]
            )
        posts = pd.DataFrame(
            posts,
            columns=[
                "title",
                "score",
                "id",
                "url",
                "reddit_url",
                "num_comments",
                "body",
                "scrape_date",
                "scrape_time",
            ],
        )
        posts.columns = posts.columns.str.lower()

        logging.info(
            f"Reddit Scrape Successful, grabbing 27 Recent popular posts from r/{sub} subreddit"
        )
        return posts
    except BaseException as error:
        logging.error(f"Reddit Scrape Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        data = []
        return data


def get_reddit_comments(urls: pd.Series) -> pd.DataFrame:
    """
    Web Scrape function w/ PRAW that iteratively extracts comments from provided reddit post urls.

    Args:
        urls (Series): The (reddit) urls to extract comments from

    Returns:
        Pandas DataFrame of all comments from the provided reddit urls
    """
    reddit = praw.Reddit(
        client_id=os.environ.get("reddit_accesskey"),
        client_secret=os.environ.get("reddit_secretkey"),
        user_agent="praw-app",
        username=os.environ.get("reddit_user"),
        password=os.environ.get("reddit_pw"),
    )
    author_list = []
    comment_list = []
    score_list = []
    flair_list1 = []
    flair_list2 = []
    edited_list = []
    url_list = []

    try:
        for i in urls:
            submission = reddit.submission(url=i)
            submission.comments.replace_more(limit=0)
            # this removes all the "more comment" stubs
            # to grab ALL comments use limit=None, but it will take 100x longer
            for comment in submission.comments.list():
                author_list.append(comment.author)
                comment_list.append(comment.body)
                score_list.append(comment.score)
                flair_list1.append(comment.author_flair_css_class)
                flair_list2.append(comment.author_flair_text)
                edited_list.append(comment.edited)
                url_list.append(i)

        df = pd.DataFrame(
            {
                "author": author_list,
                "comment": comment_list,
                "score": score_list,
                "url": url_list,
                "flair1": flair_list1,
                "flair2": flair_list2,
                "edited": edited_list,
                "scrape_date": datetime.now().date(),
                "scrape_ts": datetime.now(),
            }
        )

        df = df.astype({"author": str})
        df = df.query('author != "None"')  # remove deleted comments rip
        df = (
            df.sort_values("score").groupby(["author", "comment", "url"]).tail(1)
        )  # remove duplicates, grab comment with highest score
        # adding sentiment analysis columns
        df = add_sentiment_analysis(df, "comment")

        df["edited"] = np.where(
            df["edited"] == False, 0, 1
        )  # if edited, then 1, else 0
        df["md5_pk"] = df.apply(
            lambda x: hashlib.md5(
                (str(x["author"]) + str(x["comment"]) + str(x["url"])).encode("utf8")
            ).hexdigest(),
            axis=1,
        )
        # this hash function lines up with the md5 fucntion in postgres
        logging.info(
            f"Reddit Comment Extraction Success, retrieving {len(df)} total comments from {len(urls)} total urls"
        )
        return df
    except BaseException as e:
        logging.error(f"Reddit Comment Extraction Failed for url {i}, {e}")
        sentry_sdk.capture_exception(e)
        df = []
        return df


def scrape_tweets_tweepy(
    search_parameter: str, count: int, result_type: str
) -> pd.DataFrame:
    """
    Web Scrape function w/ Tweepy to scrape Tweets made within last ~ 7 days

    Args:
        search_parameter (str): The string you're interested in finding Tweets for

        count (int): Number of tweets to grab

        result_type (str): Either mixed, recent, or popular.

    Returns:
        Pandas DataFrame of recent Tweets
    """
    auth = tweepy.OAuthHandler(
        os.environ.get("twitter_consumer_api_key"),
        os.environ.get("twitter_consumer_api_secret"),
    )

    # auth.set_access_token(
    #     os.environ.get("twitter_access_api_key"),
    #     os.environ.get("twitter_access_api_secret"),
    # )

    api = tweepy.API(auth, wait_on_rate_limit=True)

    df = pd.DataFrame()
    try:
        for tweet in tweepy.Cursor(  # result_type can be mixed, recent, or popular.
            api.search_tweets, search_parameter, count=count, result_type=result_type
        ).items(count):
            # print(status)
            df = df.append(
                {
                    "created_at": tweet._json["created_at"],
                    "tweet_id": tweet._json["id_str"],
                    "username": tweet._json["user"]["screen_name"],
                    "user_id": tweet._json["user"]["id"],
                    "tweet": tweet._json["text"],
                    "likes": tweet._json["favorite_count"],
                    "retweets": tweet._json["retweet_count"],
                    "language": tweet._json["lang"],
                    "scrape_ts": datetime.now(),
                    "profile_img": tweet._json["user"]["profile_image_url"],
                    "url": f"https://twitter.com/twitter/statuses/{tweet._json['id']}",
                },
                ignore_index=True,
            )

        df = add_sentiment_analysis(df, "tweet")
        logging.info(f"Twitter Scrape Successful, retrieving {len(df)} Tweets")
        return df
    except BaseException as e:
        logging.error(f"Error Occurred for Scrape Tweets Tweepy, {e}")
        sentry_sdk.capture_exception(e)
        df = []
        return df


def scrape_tweets_combo() -> pd.DataFrame:
    """
    Web Scrape function to scrape Tweepy Tweets for both popular & mixed tweets

    Args:
        None

    Returns:
        Pandas DataFrame of both popular and mixed tweets.
    """
    try:
        df1 = scrape_tweets_tweepy("nba", 1000, "popular")
        df2 = scrape_tweets_tweepy("nba", 5000, "mixed")

        # so the scrape_ts column screws up with filtering duplicates out so
        # this code ignores that column to correctly drop the duplicates
        df_combo = pd.concat([df1, df2])
        df_combo = df_combo.drop_duplicates(
            subset=df_combo.columns.difference(
                ["scrape_ts", "likes", "retweets", "tweet"]
            )
        )

        logging.info(
            f"Grabbing {len(df1)} Popular Tweets and {len(df2)} Mixed Tweets for {len(df_combo)} Total, {(len(df1) + len(df2) - len(df_combo))} were duplicates"
        )
        return df_combo
    except BaseException as e:
        logging.error(f"Error Occurred for Scrape Tweets Combo, {e}")
        sentry_sdk.capture_exception(e)
        df = []
        return df


def get_pbp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Web Scrape function w/ pandas read_html that uses aliases via boxscores function
    to scrape the pbp data iteratively for each game played the previous day.
    It assumes there is a location column in the df being passed in.

    Args:
        df (DataFrame) - The Boxscores DataFrame

    Returns:
        All PBP Data for the games in the input df

    """
    game_date = df['date'][0]
    try:
        if len(df) > 0:
            yesterday_hometeams = (
                df.query('location == "H"')[["team"]].drop_duplicates().dropna()
            )
            yesterday_hometeams["team"] = yesterday_hometeams["team"].str.replace(
                "PHX", "PHO"
            )
            yesterday_hometeams["team"] = yesterday_hometeams["team"].str.replace(
                "CHA", "CHO"
            )
            yesterday_hometeams["team"] = yesterday_hometeams["team"].str.replace(
                "BKN", "BRK"
            )

            away_teams = (
                df.query('location == "A"')[["team", "opponent"]]
                .drop_duplicates()
                .dropna()
            )
            away_teams = away_teams.rename(
                columns={
                    away_teams.columns[0]: "AwayTeam",
                    away_teams.columns[1]: "HomeTeam",
                }
            )
        else:
            yesterday_hometeams = []

        if len(yesterday_hometeams) > 0:
            try:
                newdate = str(
                    df["date"].drop_duplicates()[0].date()
                )  # this assumes all games in the boxscores df are 1 date
                newdate = pd.to_datetime(newdate).strftime(
                    "%Y%m%d"
                )  # formatting into url format.
                pbp_list = pd.DataFrame()
                for i in yesterday_hometeams["team"]:
                    url = f"https://www.basketball-reference.com/boxscores/pbp/{newdate}0{i}.html"
                    df = pd.read_html(url)[0]
                    df.columns = df.columns.map("".join)
                    df = df.rename(
                        columns={
                            df.columns[0]: "Time",
                            df.columns[1]: "descriptionPlayVisitor",
                            df.columns[2]: "AwayScore",
                            df.columns[3]: "Score",
                            df.columns[4]: "HomeScore",
                            df.columns[5]: "descriptionPlayHome",
                        }
                    )
                    conditions = [
                        (
                            df["HomeScore"].str.contains("Jump ball:", na=False)
                            & df["Time"].str.contains("12:00.0")
                        ),
                        (
                            df["HomeScore"].str.contains(
                                "Start of 2nd quarter", na=False
                            )
                        ),
                        (
                            df["HomeScore"].str.contains(
                                "Start of 3rd quarter", na=False
                            )
                        ),
                        (
                            df["HomeScore"].str.contains(
                                "Start of 4th quarter", na=False
                            )
                        ),
                        (
                            df["HomeScore"].str.contains(
                                "Start of 1st overtime", na=False
                            )
                        ),
                        (
                            df["HomeScore"].str.contains(
                                "Start of 2nd overtime", na=False
                            )
                        ),
                        (
                            df["HomeScore"].str.contains(
                                "Start of 3rd overtime", na=False
                            )
                        ),
                        (
                            df["HomeScore"].str.contains(
                                "Start of 4th overtime", na=False
                            )
                        ),  # if more than 4 ots then rip
                    ]
                    values = [
                        "1st Quarter",
                        "2nd Quarter",
                        "3rd Quarter",
                        "4th Quarter",
                        "1st OT",
                        "2nd OT",
                        "3rd OT",
                        "4th OT",
                    ]
                    df["Quarter"] = np.select(conditions, values, default=None)
                    df["Quarter"] = df["Quarter"].fillna(method="ffill")
                    df = df.query(
                        'Time != "Time" & Time != "2nd Q" & Time != "3rd Q" & Time != "4th Q" & Time != "1st OT" & Time != "2nd OT" & Time != "3rd OT" & Time != "4th OT"'
                    ).copy()  # use COPY to get rid of the fucking goddamn warning bc we filtered stuf out
                    # anytime you filter out values w/o copying and run code like the lines below it'll throw a warning.
                    df["HomeTeam"] = i
                    df["HomeTeam"] = df["HomeTeam"].str.replace("PHO", "PHX")
                    df["HomeTeam"] = df["HomeTeam"].str.replace("CHO", "CHA")
                    df["HomeTeam"] = df["HomeTeam"].str.replace("BRK", "BKN")
                    df = df.merge(away_teams)
                    df[["scoreAway", "scoreHome"]] = df["Score"].str.split(
                        "-", expand=True, n=1
                    )
                    df["scoreAway"] = pd.to_numeric(df["scoreAway"], errors="coerce")
                    df["scoreAway"] = df["scoreAway"].fillna(method="ffill")
                    df["scoreAway"] = df["scoreAway"].fillna(0)
                    df["scoreHome"] = pd.to_numeric(df["scoreHome"], errors="coerce")
                    df["scoreHome"] = df["scoreHome"].fillna(method="ffill")
                    df["scoreHome"] = df["scoreHome"].fillna(0)
                    df["marginScore"] = df["scoreHome"] - df["scoreAway"]
                    df["Date"] = game_date
                    df["scrape_date"] = datetime.now().date()
                    df = df.rename(
                        columns={
                            df.columns[0]: "timeQuarter",
                            df.columns[6]: "numberPeriod",
                        }
                    )
                    pbp_list = pbp_list.append(df)
                    df = pd.DataFrame()
                pbp_list.columns = pbp_list.columns.str.lower()
                pbp_list = pbp_list.query(
                    "(awayscore.notnull()) | (homescore.notnull())", engine="python"
                )
                logging.info(
                    f"PBP Data Transformation Function Successful, retrieving {len(pbp_list)} rows for {datetime.now().date()}"
                )
                # filtering only scoring plays here, keep other all other rows in future for lineups stuff etc.
                return pbp_list
            except BaseException as error:
                logging.error(f"PBP Transformation Function Logic Failed, {error}")
                sentry_sdk.capture_exception(error)
                df = []
                return df
        else:
            df = []
            logging.warning(
                f"PBP Transformation Function Failed, no data available for {datetime.now().date()}"
            )
            return df
    except BaseException as error:
        logging.error(f"PBP Data Transformation Function Failed, {error}")
        sentry_sdk.capture_exception(error)
        data = []
        return data


def schedule_scraper(
    year: str,
    month_list: List[str] = [
        "october",
        "november",
        "december",
        "january",
        "february",
        "march",
        "april",
    ],
) -> pd.DataFrame:
    """
    Web Scrape Function to scrape Schedule data by iterating through a list of months

    Args:
        year (str) - The year to scrape

        month_list (list) - List of full-month names to scrape

    Returns:
        DataFrame of Schedule Data to be stored.

    """
    try:
        schedule_df = pd.DataFrame()
        completed_months = []
        for i in month_list:
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{i}.html"
            html = requests.get(url).content
            soup = BeautifulSoup(html, "html.parser")

            headers = [th.getText() for th in soup.findAll("tr")[0].findAll("th")]
            headers[6] = "boxScoreLink"
            headers[7] = "isOT"
            headers = headers[1:]

            rows = soup.findAll("tr")[1:]
            date_info = [
                [th.getText() for th in rows[i].findAll("th")] for i in range(len(rows))
            ]

            game_info = [
                [td.getText() for td in rows[i].findAll("td")] for i in range(len(rows))
            ]
            date_info = [i[0] for i in date_info]

            schedule = pd.DataFrame(game_info, columns=headers)
            schedule["Date"] = date_info

            logging.info(
                f"Schedule Function Completed for {i}, retrieving {len(schedule)} rows"
            )
            completed_months.append(i)
            schedule_df = schedule_df.append(schedule)

        schedule_df = schedule_df[
            ["Start (ET)", "Visitor/Neutral", "Home/Neutral", "Date"]
        ]
        schedule_df["proper_date"] = pd.to_datetime(schedule_df["Date"]).dt.date
        schedule_df.columns = schedule_df.columns.str.lower()
        schedule_df = schedule_df.rename(
            columns={
                "start (et)": "start_time",
                "visitor/neutral": "away_team",
                "home/neutral": "home_team",
            }
        )

        logging.info(
            f"Schedule Function Completed for {' '.join(completed_months)}, retrieving {len(schedule_df)} total rows"
        )
        return schedule_df
    except IndexError as index_error:
        logging.info(
            f"{i} currently has no data in basketball-reference, stopping the function and returning data for {' '.join(completed_months)}"
        )
        schedule_df = schedule_df[
            ["Start (ET)", "Visitor/Neutral", "Home/Neutral", "Date"]
        ]
        schedule_df["proper_date"] = pd.to_datetime(schedule_df["Date"]).dt.date
        schedule_df.columns = schedule_df.columns.str.lower()
        schedule_df = schedule_df.rename(
            columns={
                "start (et)": "start_time",
                "visitor/neutral": "away_team",
                "home/neutral": "home_team",
            }
        )
        return schedule_df
    except BaseException as e:
        logging.error(f"Schedule Scraper Function Failed, {e}")
        df = []
        return df


def write_to_s3(
    file_name: str, df: pd.DataFrame, bucket: str = os.environ.get("S3_BUCKET")
) -> None:
    """
    S3 Function using awswrangler to write file.  Only supports parquet right now.

    Args:
        file_name (str): The base name of the file (boxscores, opp_stats)

        df (pd.DataFrame): The Pandas DataFrame to write

        bucket (str): The Bucket to write to.  Defaults to `os.environ.get('S3_BUCKET')`

    Returns:
        Writes the Pandas DataFrame to an S3 File.

    """
    month_prefix = get_leading_zeroes(datetime.now().month)
    # df['file_name'] = f'{file_name}-{datetime.now().date()}.parquet'
    try:
        if len(df) == 0:
            logging.info(f"Not storing {file_name} to s3 because it's empty.")
            pass
        elif df.schema == "Validated":
            wr.s3.to_parquet(
                df=df,
                # 2022-06-21 - use this updated s3 naming convention next season
                # f"s3://{bucket}/{file_name}/validated/year={datetime.now().year}/month={month_prefix}/{file_name}-{today}.parquet"
                path=f"s3://{bucket}/{file_name}/validated/year={datetime.now().year}/month={month_prefix}/{file_name}-{datetime.now().date()}.parquet",
                index=False,
            )
            logging.info(
                f"Storing {len(df)} {file_name} rows to S3 (s3://{bucket}/{file_name}/validated/{month_prefix}/{file_name}-{datetime.now().date()}.parquet)"
            )
            pass
        else:
            wr.s3.to_parquet(
                df=df,
                path=f"s3://{bucket}/{file_name}/invalidated/year={datetime.now().year}/month={month_prefix}/{file_name}-{datetime.now().date()}.parquet",
                index=False,
            )
            logging.info(
                f"Storing {len(df)} {file_name} rows to S3 (s3://{bucket}/{file_name}/invalidated/{month_prefix}/{file_name}-{datetime.now().date()}.parquet)"
            )
            pass
    except BaseException as error:
        logging.error(f"S3 Storage Function Failed {file_name}, {error}")
        sentry_sdk.capture_exception(error)
        pass


def write_to_sql(con, table_name: str, df: pd.DataFrame, table_type: str) -> None:
    """
    SQL Table function to write a pandas data frame in aws_dfname_source format

    Args:
        con (SQL Connection): The connection to the SQL DB.

        table_name (str): The Table name to write to SQL as.

        df (DataFrame): The Pandas DataFrame to store in SQL

        table_type (str): Whether the table should replace or append to an existing SQL Table under that name

    Returns:
        Writes the Pandas DataFrame to a Table in the Schema we connected to.

    """
    try:
        if len(df) == 0:
            logging.info(f"{table_name} is empty, not writing to SQL")
        elif df.schema == "Validated":
            df.to_sql(
                con=con,
                name=f"aws_{table_name}_source",
                index=False,
                if_exists=table_type,
            )
            logging.info(
                f"Writing {len(df)} {table_name} rows to aws_{table_name}_source to SQL"
            )
        else:
            logging.info(f"{table_name} Schema Invalidated, not writing to SQL")
    except BaseException as error:
        logging.error(f"SQL Write Script Failed, {error}")
        sentry_sdk.capture_exception(error)


def write_to_sql_upsert(
    conn, table_name: str, df: pd.DataFrame, table_type: str, pd_index: List[str]
) -> None:
    """
    SQL Table function to upsert a Pandas DataFrame into a SQL Table.

    Will create a new table if it doesn't exist.  If it does, it will insert new records and upsert new column values onto existing records (if applicable).

    You have to do some extra index stuff to the pandas df to specify what the primary key of the records is (this data does not get upserted).

    Args:
        conn (SQL Connection): The connection to the SQL DB.

        table_name (str): The Table name to write to SQL as.

        df (DataFrame): The Pandas DataFrame to store in SQL

        table_type (str): A placeholder which should always be "upsert"

        pd_index (List[str]): The columns that make up the composite primary key of the SQL Table.

    Returns:
        Upserts any new data in the Pandas DataFrame to the table in Postgres in the {nba_source_dev} schema

    """
    sql_table_name = f"aws_{table_name}_source"
    if len(df) == 0:
        logging.info(f"{sql_table_name} is empty, not storing to SQL")
        pass
    else:
        # 2 try except blocks bc in event of an error there needs to be different logic to safely exit out and continue script
        try:
            df = df.set_index(pd_index)
            df = df.rename_axis(pd_index)

            if not conn.execute(
                f"""SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE  table_schema = 'nba_source' 
                    AND    table_name   = '{sql_table_name}');
                    """
            ).first()[0]:
                # If the table does not exist, we should just use to_sql to create it
                df.to_sql(sql_table_name, conn)
                print(
                    f"SQL Upsert Function Successful, {len(df)} records added to a NEW TABLE {sql_table_name}"
                )
                pass
        except BaseException as error:
            sentry_sdk.capture_exception(error)
            logging.error(
                f"SQL Upsert Function Failed for NEW TABLE {sql_table_name} ({len(df)} rows), {error}"
            )
            pass
        else:
            try:
                # If it already exists...
                temp_table_name = f"temp_{uuid.uuid4().hex[:6]}"
                df.to_sql(temp_table_name, conn, index=True)

                index = list(df.index.names)
                index_sql_txt = ", ".join([f'"{i}"' for i in index])
                columns = list(df.columns)
                headers = index + columns
                headers_sql_txt = ", ".join([f'"{i}"' for i in headers])
                # this is excluding the primary key columns needed to identify the unique rows.
                update_column_stmt = ", ".join(
                    [f'"{col}" = EXCLUDED."{col}"' for col in columns]
                )

                # For the ON CONFLICT clause, postgres requires that the columns have unique constraint
                query_pk = f"""
                ALTER TABLE "{sql_table_name}" DROP CONSTRAINT IF EXISTS unique_constraint_for_upsert_{table_name};
                ALTER TABLE "{sql_table_name}" ADD CONSTRAINT unique_constraint_for_upsert_{table_name} UNIQUE ({index_sql_txt});
                """

                conn.execute(query_pk)

                # Compose and execute upsert query
                query_upsert = f"""
                INSERT INTO "{sql_table_name}" ({headers_sql_txt}) 
                SELECT {headers_sql_txt} FROM "{temp_table_name}"
                ON CONFLICT ({index_sql_txt}) DO UPDATE 
                SET {update_column_stmt};
                """
                conn.execute(query_upsert)
                conn.execute(f"DROP TABLE {temp_table_name};")
                logging.info(
                    f"SQL Upsert Function Successful, {len(df)} records added or upserted into {table_name}"
                )
                pass
            except BaseException as error:
                conn.execute(f"DROP TABLE {temp_table_name};")
                sentry_sdk.capture_exception(error)
                logging.error(
                    f"SQL Upsert Function Failed for EXISTING {table_name} ({len(df)} rows), {error}"
                )
                pass


def sql_connection(
    rds_schema: str,
    RDS_USER: str = os.environ.get("RDS_USER"),
    RDS_PW: str = os.environ.get("RDS_PW"),
    RDS_IP: str = os.environ.get("IP"),
    RDS_DB: str = os.environ.get("RDS_DB"),
) -> Engine:
    """
    SQL Connection function to define the SQL Driver + connection variables needed to connect to the DB.
    This doesn't actually make the connection, use conn.connect() in a context manager to create 1 re-usable connection

    Args:
        rds_schema (str): The Schema in the DB to connect to.

    Returns:
        SQL Connection variable to a specified schema in my PostgreSQL DB
    """
    try:
        connection = create_engine(
            f"postgresql+psycopg2://{RDS_USER}:{RDS_PW}@{RDS_IP}:5432/{RDS_DB}",
            connect_args={"options": f"-csearch_path={rds_schema}"},
            # defining schema to connect to
            echo=False,
        )
        logging.info(f"SQL Connection to schema: {rds_schema} Successful")
        return connection
    except exc.SQLAlchemyError as e:
        logging.error(f"SQL Connection to schema: {rds_schema} Failed, Error: {e}")
        sentry_sdk.capture_exception(e)
        return e


def send_aws_email(logs: pd.DataFrame) -> None:
    """
    Email function utilizing boto3, has to be set up with SES in AWS and env variables passed in via Terraform.
    The actual email code is copied from aws/boto3 and the subject / message should go in the subject / body_html variables.

    Args:
        logs (DataFrame): The log file name generated by the script.

    Returns:
        Sends an email out upon every script execution, including errors (if any)
    """
    sender = os.environ.get("USER_EMAIL")
    recipient = os.environ.get("USER_EMAIL")
    aws_region = "us-east-1"
    subject = f"NBA ELT PIPELINE - {str(len(logs))} Alert Fails for {str(datetime.now().date())}"
    body_html = message = f"""\
<h3>Errors:</h3>
                   {logs.to_html()}"""

    charset = "UTF-8"
    client = boto3.client("ses", region_name=aws_region)
    try:
        response = client.send_email(
            Destination={"ToAddresses": [recipient,],},
            Message={
                "Body": {
                    "Html": {"Charset": charset, "Data": body_html,},
                    "Text": {"Charset": charset, "Data": body_html,},
                },
                "Subject": {"Charset": charset, "Data": subject,},
            },
            Source=sender,
        )
    except ClientError as e:
        logging.error(e.response["Error"]["Message"])
    else:
        logging.info("Email sent! Message ID:"),
        logging.info(response["MessageId"])


# DEPRECATING this as of 2022-04-25 - i send emails everyday now regardless of pass or fail
def execute_email_function(logs: pd.DataFrame) -> None:
    """
    Email function that executes the email function upon script finishing.
    This is really not necessary; originally thought i wouldn't email if no errors would found but now i send it everyday regardless.

    Args:
        logs (DataFrame): The log file name generated by the script.

    Returns:
        Holds the actual send_email logic and executes if invoked as a script (aka on ECS)
    """
    try:
        if len(logs) > 0:
            logging.info("Sending Email")
            send_aws_email(logs)
        elif len(logs) == 0:
            logging.info("No Errors!")
            send_aws_email(logs)
    except BaseException as error:
        logging.error(f"Failed Email Alert, {error}")
        sentry_sdk.capture_exception(error)
