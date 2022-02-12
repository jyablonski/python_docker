import numpy as np

stats_schema = {
    "player": np.dtype("O"),
    "pos": np.dtype("O"),
    "age": np.dtype("float64"),
    "tm": np.dtype("O"),
    "g": np.dtype("float64"),
    "gs": np.dtype("float64"),
    "mp": np.dtype("float64"),
    "fg": np.dtype("float64"),
    "fga": np.dtype("float64"),
    "fg%": np.dtype("float64"),
    "3p": np.dtype("float64"),
    "3pa": np.dtype("float64"),
    "3p%": np.dtype("float64"),
    "2p": np.dtype("float64"),
    "2pa": np.dtype("float64"),
    "2p%": np.dtype("float64"),
    "efg%": np.dtype("float64"),
    "ft": np.dtype("float64"),
    "fta": np.dtype("float64"),
    "ft%": np.dtype("float64"),
    "orb": np.dtype("float64"),
    "drb": np.dtype("float64"),
    "trb": np.dtype("float64"),
    "ast": np.dtype("float64"),
    "stl": np.dtype("float64"),
    "blk": np.dtype("float64"),
    "tov": np.dtype("float64"),
    "pf": np.dtype("float64"),
    "pts": np.dtype("float64"),
    "scrape_date": np.dtype("O"),
}

adv_stats_schema = {
    "index": np.dtype("int64"),
    "team": np.dtype("O"),
    "age": np.dtype("O"),
    "w": np.dtype("O"),
    "l": np.dtype("O"),
    "pw": np.dtype("O"),
    "pl": np.dtype("O"),
    "mov": np.dtype("O"),
    "sos": np.dtype("O"),
    "srs": np.dtype("O"),
    "ortg": np.dtype("O"),
    "drtg": np.dtype("O"),
    "nrtg": np.dtype("O"),
    "pace": np.dtype("O"),
    "ftr": np.dtype("O"),
    "3par": np.dtype("O"),
    "ts%": np.dtype("O"),
    "efg%": np.dtype("O"),
    "tov%": np.dtype("O"),
    "orb%": np.dtype("O"),
    "ft/fga": np.dtype("O"),
    "efg%_opp": np.dtype("O"),
    "tov%_opp": np.dtype("O"),
    "drb%_opp": np.dtype("O"),
    "ft/fga_opp": np.dtype("O"),
    "arena": np.dtype("O"),
    "attendance": np.dtype("O"),
    "att/game": np.dtype("O"),
    "scrape_date": np.dtype("O"),
}

boxscores_schema = {
    "player": np.dtype("O"),
    "team": np.dtype("O"),
    "location": np.dtype("O"),
    "opponent": np.dtype("O"),
    "outcome": np.dtype("O"),
    "mp": np.dtype("O"),
    "fgm": np.dtype("float64"),
    "fga": np.dtype("float64"),
    "fgpercent": np.dtype("float64"),
    "threepfgmade": np.dtype("float64"),
    "threepattempted": np.dtype("float64"),
    "threepointpercent": np.dtype("float64"),
    "ft": np.dtype("float64"),
    "fta": np.dtype("float64"),
    "ftpercent": np.dtype("float64"),
    "oreb": np.dtype("float64"),
    "dreb": np.dtype("float64"),
    "trb": np.dtype("float64"),
    "ast": np.dtype("float64"),
    "stl": np.dtype("float64"),
    "blk": np.dtype("float64"),
    "tov": np.dtype("float64"),
    "pf": np.dtype("float64"),
    "pts": np.dtype("float64"),
    "plusminus": np.dtype("float64"),
    "gmsc": np.dtype("float64"),
    "date": np.dtype("<M8[ns]"),
    "type": np.dtype("O"),
    "season": np.dtype("int64"),
}

boxscores_schema_fake = {
    "player": np.dtype("O"),
    "team": np.dtype("O"),
    "location": np.dtype("O"),
    "opponent": np.dtype("O"),
    "outcome": np.dtype("O"),
    "mp": np.dtype("O"),
    "fgm": np.dtype("float64"),
    "fga": np.dtype("float64"),
    "fgpercent": np.dtype("float64"),
    "threepfgmade": np.dtype("float64"),
    "threepattempted": np.dtype("float64"),
    "threepointpercent": np.dtype("float64"),
    "ft": np.dtype("float64"),
    "fta": np.dtype("float64"),
    "ftpercent": np.dtype("float64"),
    "oreb": np.dtype("float64"),
    "dreb": np.dtype("float64"),
    "trb": np.dtype("float64"),
    "ast": np.dtype("float64"),
    "stl": np.dtype("float64"),
    "blk": np.dtype("float64"),
    "tov": np.dtype("float64"),
    "pf": np.dtype("float64"),
    "pts": np.dtype("float64"),
    "plusminus": np.dtype("float64"),
    "gmsc": np.dtype("float64"),
    "date": np.dtype("<M8[ns]"),
    "type": np.dtype("O"),
    "season": np.dtype("int64"),
    "FAKE_COLUMN": np.dtype("O"),
}

injury_schema = {
    "player": np.dtype("O"),
    "team": np.dtype("O"),
    "date": np.dtype("O"),
    "description": np.dtype("O"),
    "scrape_date": np.dtype("O"),
}

opp_stats_schema = {
    "team": np.dtype("O"),
    "fg_percent_opp": np.dtype("float64"),
    "threep_percent_opp": np.dtype("float64"),
    "threep_made_opp": np.dtype("float64"),
    "ppg_opp": np.dtype("float64"),
    "scrape_date": np.dtype("O"),
}


pbp_data_schema = {
    "timequarter": np.dtype("O"),
    "descriptionplayvisitor": np.dtype("O"),
    "awayscore": np.dtype("O"),
    "score": np.dtype("O"),
    "homescore": np.dtype("O"),
    "descriptionplayhome": np.dtype("O"),
    "numberperiod": np.dtype("O"),
    "hometeam": np.dtype("O"),
    "awayteam": np.dtype("O"),
    "scoreaway": np.dtype("float64"),
    "scorehome": np.dtype("float64"),
    "marginscore": np.dtype("float64"),
    "date": np.dtype("O"),
}

reddit_data_schema = {
    "title": np.dtype("O"),
    "score": np.dtype("int64"),
    "id": np.dtype("O"),
    "url": np.dtype("O"),
    "reddit_url": np.dtype("O"),
    "num_comments": np.dtype("int64"),
    "body": np.dtype("O"),
    "scrape_date": np.dtype("O"),
    "scrape_time": np.dtype("<M8[ns]"),
}

reddit_comment_data_schema = {
    "author": np.dtype("O"),
    "comment": np.dtype("O"),
    "score": np.dtype("int64"),
    "url": np.dtype("O"),
    "flair1": np.dtype("O"),
    "flair2": np.dtype("O"),
    "edited": np.dtype("O"),
    "scrape_date": np.dtype("O"),
    "scrape_ts": np.dtype("<M8[ns]"),
    "compound": np.dtype("float64"),
    "neg": np.dtype("float64"),
    "neu": np.dtype("float64"),
    "pos": np.dtype("float64"),
    "sentiment": np.dtype("int64"),
}

odds_schema = {
    "team": np.dtype("O"),
    "spread": np.dtype("O"),
    "total": np.dtype("O"),
    "moneyline": np.dtype("int64"),
    "date": np.dtype("O"),
    "datetime1": np.dtype("<M8[ns]"),
}

transactions_schema = {
    "date": np.dtype("<M8[ns]"),
    "transaction": np.dtype("O"),
    "scrape_date": np.dtype("O"),
}

twitter_data_schema = {
    "created_at": np.dtype("O"),
    "date": np.dtype("O"),
    "username": np.dtype("O"),
    "tweet": np.dtype("O"),
    "language": np.dtype("O"),
    "link": np.dtype("O"),
    "likes_count": np.dtype("int64"),
    "retweets_count": np.dtype("int64"),
    "replies_count": np.dtype("int64"),
    "scrape_date": np.dtype("O"),
    "scrape_ts": np.dtype("<M8[ns]"),
    "compound": np.dtype("float64"),
    "neg": np.dtype("float64"),
    "neu": np.dtype("float64"),
    "pos": np.dtype("float64"),
    "sentiment": np.dtype("int64"),
}
