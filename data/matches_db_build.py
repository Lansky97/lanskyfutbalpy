import soccerdata as sd
import pandas as pd
import numpy as np
import duckdb
import argparse
from typing import Optional
from datetime import datetime, timezone

ALL_LEAGUES = ['Big 5 European Leagues Combined',
 'ENG-Championship',
 'ESP-La Liga 2',
 'FRA-Ligue 2',
 'GER-2. Bundesliga',
 'ITA-Serie B']

LEAGUE_INFO = {
    'Premier_League':   {'tier': 1, 'country': 'England'},
    'La_Liga':          {'tier': 1, 'country': 'Spain'},
    'Bundesliga':       {'tier': 1, 'country': 'Germany'},
    'Serie_A':          {'tier': 1, 'country': 'Italy'},
    'Ligue_1':          {'tier': 1, 'country': 'France'},
    'Championship':     {'tier': 2, 'country': 'England'},
    'La_Liga_2':        {'tier': 2, 'country': 'Spain'},
    '2._Bundesliga':    {'tier': 2, 'country': 'Germany'},
    'Serie_B':          {'tier': 2, 'country': 'Italy'},
    'Ligue_2':          {'tier': 2, 'country': 'France'}
}

SCHEMA_COLUMNS = [
    "season_end_year",
    "country",
    "tier",
    "league",
    "date",
    "time",
    "home_team",
    "home_xg",
    "home_g",
    "away_g",
    "away_xg",
    "away_team",
    "status",
    "match_id",
    "game_id",
    "season",
    "round",
    "week",
    "day",
    "venue",
    "attendance",
    "referee",
    "match_report",
    "notes",
    "source",
    "last_scraped_at"
    ]

def normalise_league_name(matches: pd.DataFrame) -> pd.DataFrame:
    s = matches['league'].fillna("").astype(str)
    parts = s.str.split("-", n=1, expand=True)

    if parts.shape[1] > 1:
        league_names = parts[1].fillna(s).str.strip().str.replace(" ", "_")
        matches['league'] = league_names
    
    matches['country'] = matches['league'].map(lambda x: LEAGUE_INFO.get(x, {}).get('country', ""))
    matches['tier'] = matches['league'].map(lambda x: LEAGUE_INFO.get(x, {}).get('tier', pd.NA))
    
    return matches

def split_score(matches: pd.DataFrame) -> pd.DataFrame:
    s = matches['score'].fillna("").astype(str).str.strip()
    s = s.str.replace(r"[–-]", "-", regex=True)

    score_pattern = r"(?:\(\d+\)\s*)?(\d+)\s*-\s*(\d+)(?:\s*\(\d+\))?"

    extracted = s.str.extract(score_pattern)
    matches['home_g'] = pd.to_numeric(extracted[0], errors="coerce").astype("Int64")
    matches['away_g'] = pd.to_numeric(extracted[1], errors="coerce").astype("Int64")

    for col in ['score', 'game']:
        if col in matches.columns:
            matches = matches.drop(columns=[col])
    return matches

def set_match_id(matches: pd.DataFrame) -> pd.DataFrame:
    playoff_keywords = 'play[- ]?offs?|play[- ]?outs?|tie[- ]?breaker'
    play_offs_mask = matches['round'].astype(str).str.contains(playoff_keywords, case=False, na=False)
    
    home_team_clean = matches['home_team'].astype(str).str.replace(" ", "_")
    away_team_clean = matches['away_team'].astype(str).str.replace(" ", "_")
    
    match_id = (
        matches['league'].astype(str) + '_' +
        matches['season'].astype(str) + '_' +
        home_team_clean + '_' +
        away_team_clean
    )
    match_id_with_round = (
        matches['league'].astype(str) + '_' +
        matches['season'].astype(str) + '_' +
        home_team_clean + '_' +
        away_team_clean + '_' +
        "playoffs"
    )
    matches['match_id'] = np.where(play_offs_mask, match_id_with_round, match_id)
    return matches

def infer_status(matches: pd.DataFrame) -> pd.DataFrame:
    notes = matches.get("notes", pd.Series([""] * len(matches)))
    notes_s = notes.fillna("").astype(str).str.lower()

    played = matches['home_g'].notna() & matches['away_g'].notna()
    postponed = ~played & notes_s.str.contains("postpon", na=False)
    cancelled = ~played & notes_s.str.contains("cancel", na=False)
    abandoned = ~played & notes_s.str.contains("abandon", na=False)

    conditions = [played, postponed, cancelled, abandoned]
    choices = ["played", "postponed", "cancelled", "abandoned"]

    matches["status"] = np.select(conditions, choices, "scheduled")
    return matches

def _coerce_time(col: pd.Series) -> pd.Series:
    if col is None:
        return pd.Series([pd.NaT] * 0) 
    t = pd.to_datetime(col, errors="coerce").dt.time
    return t

def _nullable_float(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col, errors="coerce").astype("Float64")

def _nullable_int(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col, errors="coerce").astype("Int64")

def _safe_col(df: pd.DataFrame, name: str, default, dtype=None):
    s = df[name] if name in df.columns else pd.Series([default] * len(df))
    if dtype == "float":
        return _nullable_float(s)
    if dtype == "int":
        return _nullable_int(s)
    if dtype == "date":
        return _coerce_time(s)
    if dtype == "str":
        return s.astype("str")
    return s

def align_to_schema(matches: pd.DataFrame, schema_columns: list[str] = SCHEMA_COLUMNS) -> pd.DataFrame:
    matches_df = matches.copy()

    int_columns = [
        "season_end_year",
        "home_g",
        "away_g",
        "attendance"
        ]
    float_columns = [
        "home_xg",
        "away_xg"
        ]
    date_columns = [
        "date",
        "last_scraped_at"
        ]

    for col in int_columns:
        if col in int_columns:
            matches_df[col] = _safe_col(matches_df, col, default=pd.NA, dtype="int")
        elif col in float_columns:
            matches_df[col] = _safe_col(matches_df, col, default=pd.NA, dtype="float")
        elif col in date_columns:
            matches_df[col] = _safe_col(matches_df, col, default=pd.NaT, dtype="date")
        else:
            matches_df[col] = _safe_col(matches_df, col, default="", dtype="str")
    return matches_df[schema_columns]

def scrape_FBref_matches(
        competition: list[str] | None,
        seasons: list[int] | list[str]
    ) -> pd.DataFrame:

    leagues = competition if competition else ALL_LEAGUES
    fbref = sd.FBref(leagues=leagues, seasons=seasons)
    raw_matches = fbref.read_schedule().reset_index()

    processed_matches = (
        raw_matches
        .pipe(normalise_league_name)
        .pipe(split_score)
        .pipe(set_match_id)
        .pipe(infer_status)
    )
    processed_matches['source'] = "fbref"
    processed_matches['season_end_year'] = processed_matches['season'].apply(lambda x: 2000 + int(str(x)[-2:]))
    processed_matches['last_scraped_at'] = pd.Timestamp.now(timezone.utc)

    processed_matches = align_to_schema(processed_matches)
    return processed_matches

# ------------- DUCKDB -------------
DB_LOAD = """
CREATE TABLE IF NOT EXISTS matches (
    season_end_year INTEGER NOT NULL,
    country TEXT NOT NULL,
    tier INTEGER,
    league TEXT NOT NULL,
    date DATE NOT NULL,
    time TIME,
    home_team TEXT NOT NULL,
    home_xg FLOAT,
    home_g INTEGER,
    away_g INTEGER,
    away_xg FLOAT,
    away_team TEXT NOT NULL,
    status TEXT NOT NULL,
    match_id TEXT PRIMARY KEY,
    game_id TEXT,
    season TEXT NOT NULL,
    round TEXT,
    week TEXT,
    day TEXT,
    venue TEXT,
    attendance INTEGER,
    referee TEXT,
    match_report TEXT,
    notes TEXT,
    source TEXT NOT NULL,
    last_scraped_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS staging_matches AS
SELECT * FROM matches WHERE 1=0;

CREATE VIEW IF NOT EXISTS ratings AS
WITH home_stats AS (
    SELECT
        season_end_year,
        country,
        tier,
        league,
        home_team AS team, 
        SUM(home_g) AS home_g,
        SUM(away_g) AS home_ga,
        SUM(home_xg) AS home_xg,
        SUM(away_xg) AS home_xga,
        SUM(
            CASE
                WHEN match_id LIKE '%playoffs%' THEN 0
                WHEN home_g > away_g THEN 3
                WHEN home_g < away_g THEN 0
                WHEN home_g = away_g THEN 1
                ELSE 0
            END
        ) AS home_pts,
        SUM(
            CASE
                WHEN status = 'played' THEN 1
                ELSE 0
            END
        ) AS home_p
    FROM matches
    WHERE match_id NOT LIKE '%playoffs%'
    GROUP BY season_end_year, country, tier, league, home_team
),
away_stats AS (
    SELECT
        season_end_year,
        country,
        tier,
        league,
        away_team AS team, 
        SUM(away_g) AS away_g,
        SUM(home_g) AS away_ga,
        SUM(away_xg) AS away_xg,
        SUM(home_xg) AS away_xga,
        SUM(
            CASE
                 WHEN match_id LIKE '%playoffs%' THEN 0
                WHEN away_g > home_g THEN 3
                WHEN away_g < home_g THEN 0
                WHEN away_g = home_g THEN 1
                ELSE 0
            END
        ) AS away_pts,
        SUM(
            CASE
                WHEN status = 'played' THEN 1
                ELSE 0
            END
        ) AS away_p
    FROM matches
    WHERE match_id NOT LIKE '%playoffs%'
    GROUP BY season_end_year, country, tier, league, away_team
),
combined_stats AS (
    SELECT
        home_stats.season_end_year,
        home_stats.country,
        home_stats.tier,
        LAG(home_stats.tier) OVER (PARTITION BY home_stats.team ORDER BY home_stats.season_end_year) AS prev_tier,
        LEAD(home_stats.tier) OVER (PARTITION BY home_stats.team ORDER BY home_stats.season_end_year) AS next_tier,
        home_stats.league,
        home_stats.team,
        home_stats.home_g,
        home_stats.home_ga,
        home_stats.home_xg,
        home_stats.home_xga,
        home_stats.home_pts,
        home_stats.home_p,
        away_stats.away_g,
        away_stats.away_ga,
        away_stats.away_xg,
        away_stats.away_xga,
        away_stats.away_pts,
        away_stats.away_p
    FROM home_stats
    JOIN away_stats USING (season_end_year, country, tier, league, team)
)
SELECT
    season_end_year,
    country,
    tier,
    prev_tier,
    next_tier,
    league,
    team,
    home_g,
    home_ga,
    home_xg,
    home_xga,
    home_pts,
    home_p,
    away_g,
    away_ga,
    away_xg,
    away_xga,
    away_pts,
    away_p,
    (home_g + away_g) AS total_g,
    (home_ga + away_ga) AS total_ga,
    (home_xg + away_xg) AS total_xg,
    (home_xga + away_xga) AS total_xga,
    (home_pts + away_pts) AS total_pts,
    (home_p + away_p) AS total_p,
    RANK() OVER (
        PARTITION BY season_end_year, league
        ORDER BY (home_pts + away_pts) DESC, (home_g + away_g - home_ga - away_ga) DESC, team ASC
    ) AS finishing_position,
    CASE
        WHEN tier = prev_tier THEN 'none'
        WHEN tier > prev_tier THEN 'relegated'
        WHEN tier < prev_tier THEN 'promoted'
        ELSE 'none'
    END AS season_start_flag,
    CASE
        WHEN tier = next_tier THEN 'none'
        WHEN tier > next_tier THEN 'promoted'
        WHEN tier < next_tier THEN 'relegated'
        ELSE 'none'
    END AS season_end_flag
FROM combined_stats
ORDER BY season_end_year, country, tier, finishing_position;

CREATE VIEW IF NOT EXISTS results AS
SELECT 
season_end_year,
league,
date,
time,
home_team,
home_g,
away_g,
away_xg,
away_team, 
match_id FROM matches WHERE status = 'played';

CREATE VIEW IF NOT EXISTS fixtures AS
SELECT
season_end_year,
league,
date,
time,
home_team,
away_team, 
match_id FROM matches WHERE status = 'scheduled';
"""

def build_upsertquery(id_col: str = "match_id") -> tuple[str, str]:
    non_id_cols = [col for col in SCHEMA_COLUMNS if col not in [id_col, 'last_scraped_at']]

    # -- UPDATE ----
    set_clauses = [f"{col} = s.{col}" for col in non_id_cols]
    diff_clauses = [f"t.{col} IS DISTINCT FROM s.{col}" for col in non_id_cols]
    update_sql = f"""
    UPDATE matches AS t
    SET {', '.join(set_clauses)}, last_scraped_at = s.last_scraped_at
    FROM staging_matches AS s
    WHERE t.{id_col} = s.{id_col}
    AND ({' OR '.join(diff_clauses)});
    """

    # -- INSERT ----
    cols = ", ".join(SCHEMA_COLUMNS)
    select_s_cols = ", ".join([f"s.{col}" for col in SCHEMA_COLUMNS])
    insert_sql = f"""
    INSERT INTO matches ({cols})
    SELECT {select_s_cols}
    FROM staging_matches AS s
    LEFT JOIN matches t ON s.{id_col} = t.{id_col}
    WHERE t.{id_col} IS NULL;
    """
    return update_sql, insert_sql

def ensure_DB_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute(DB_LOAD)

def load_to_staging(conn: duckdb.DuckDBPyConnection, matches: pd.DataFrame):
    conn.execute("DELETE FROM staging_matches")
    conn.register("staging_df", matches)
    cols = ", ".join(SCHEMA_COLUMNS)
    conn.execute(f" INSERT INTO staging_matches ({cols}) SELECT {cols} FROM staging_df;")
    conn.unregister("staging_df")

def merge_staging(conn: duckdb.DuckDBPyConnection, id_col: str = "match_id") -> None:
    UPDATE_SQL, INSERT_SQL = build_upsertquery(id_col=id_col)
    conn.execute(UPDATE_SQL)
    conn.execute(INSERT_SQL)

# ----------  CLI / ORCHESTRATION  ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/update DuckDB from soccerdata FBref scrapes")
    p.add_argument("--db", default="data/football_data.db", help="Path to DuckDB file")
    p.add_argument("--leagues", nargs="+", required=True, help="League names")
    p.add_argument("--seasons", nargs="+", required=True,
                   help="Seasons e.g. single year 1718 or range 2017-2019")
    p.add_argument("--append-only", action="store_true",
                   help="Skip updates; only insert new rows")
    args = p.parse_args()
    args.seasons = _parse_seasons(args.seasons)
    args.leagues = _parse_leagues(args.leagues)
    return args

def _parse_seasons(raw: list[str]) -> list[str]:
    out = []
    if len(raw) == 1 and "-" in raw[0]:
        start, end = map(int, raw[0].split("-"))
        if start > end:
            raise ValueError(f"Invalid season range: {raw[0]}")
        for year in range(start, end):
            out.append(f"{str(year)[-2:]}-{str(year + 1)[-2:]}")
    else:
        for year in raw:
            try:
                year = int(year)
                out.append(f"{str(year - 1)[-2:]}-{str(year)[-2:]}")
            except ValueError:
                raise ValueError(f"Invalid season format: {year}")
    return out

def _parse_leagues(raw: list[str]) -> list[str]:
    if len(raw) == 1:
        key = raw[0].lower()
        if key in ("default", "all", "top5"):
            return ALL_LEAGUES
    return raw

def main():
    args = parse_args()

    processed_matches = scrape_FBref_matches(
        competition=args.leagues,
        seasons=args.seasons
    )

    conn = duckdb.connect(args.db, read_only=False)
    ensure_DB_schema(conn)

    load_to_staging(conn, processed_matches)

    if args.append_only:
        conn.execute("""
            INSERT INTO matches
            SELECT s.* 
            FROM staging_matches AS s
            LEFT JOIN matches t USING(match_id)
            WHERE t.match_id IS NULL
        """)
    else:
        merge_staging(conn)
    
    print("Completed merge.")
    conn.close()

if __name__ == "__main__":
    main()