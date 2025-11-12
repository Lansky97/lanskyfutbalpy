import csv
import pandas as pd
import duckdb as ddb

def read_schedule(filepath = None, season_end_year = None, league = None):
    if filepath:
        return read_schedule_from_csv(filepath)
    else:
        return read_schedule_from_database(season_end_year, league)

def read_schedule_from_csv(file_path):
    results = []
    with open(file_path,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            results.append(row)
    return results

def read_schedule_from_database(season_end_year = 2025, league = 'Premier_League'):
    con = ddb.connect(database='data/football_data.db', read_only=True)
    query = """
    SELECT 
    league AS Competition_Name,
    'M' AS gender,
    country AS Country,
    season_end_year AS Season_End_Year,
    round AS Round,
    week AS Wk,
    day AS Day,
    STRFTIME('%Y-%m-%d', date) AS Date,
    time AS Time,
    home_team AS Home,
    home_g AS HomeGoals,
    home_xg AS Home_xG,
    away_team AS Away,
    away_g AS AwayGoals,
    away_xg AS Away_xG,
    attendance AS Attendance,
    venue AS Venue,
    referee AS Referee,
    notes AS Notes,
    match_report AS MatchURL
    FROM matches
    WHERE season_end_year = ? AND league = ?;
    """
    df = con.execute(query, [season_end_year, league]).fetch_df()
    con.close()
    return df.to_dict(orient='records')

def read_last_season_stats(season_end_year = 2025, country = 'England', tier = 1, xg_factor = 0.6):
    query = """
WITH avg_ratings AS (
    SELECT 
        country,
        tier,
        MEAN(home_g/home_p) as mean_home_g,
        MEAN(home_xg/home_p) as mean_home_xg,
        MEAN(away_g/away_p) as mean_away_g,
        MEAN(away_xg/away_p) as mean_away_xg
    FROM ratings
    WHERE country = '{country}' AND season_end_year = {season_end_year} - 1 AND tier = {tier}
    GROUP BY country, tier
),
factors AS (
    SELECT 
        country,
        tier AS next_tier,
        prev_tier AS tier,
        league_sf_home_gpg,
        league_sf_home_xgpg,
        league_sf_home_gapg,
        league_sf_home_xgapg,
        league_sf_away_gpg,
        league_sf_away_xgpg,
        league_sf_away_gapg,
        league_sf_away_xgapg
    FROM interleague_factors
    WHERE country = '{country}' AND tier = {tier} AND prev_tier = {tier} + 1
),
teams AS (
    SELECT 
        country,
        tier,
        next_tier,
        season_end_year,
        team,
        home_g/home_p AS home_g, 
        home_ga/home_p AS home_ga,
        home_xg/home_p AS home_xg,
        home_xga/home_p AS home_xga,
        away_g/away_p AS away_g,
        away_ga/away_p AS away_ga,
        away_xg/away_p AS away_xg,
        away_xga/away_p AS away_xga
    FROM ratings 
    WHERE season_end_year = {season_end_year} - 1 AND next_tier = {tier} AND country = '{country}'
),
joined AS (
    SELECT
    a.country,
    t.tier,
    t.next_tier,
    t.season_end_year,
    t.team,
    CASE
        WHEN t.tier = {tier} THEN t.home_g
        ELSE t.home_g * f.league_sf_home_gpg
    END AS home_g,
    CASE
        WHEN t.tier = {tier} THEN t.home_ga
        ELSE t.home_ga * f.league_sf_home_gapg
    END AS home_ga,
    CASE
        WHEN t.tier = {tier} THEN t.home_xg
        ELSE t.home_xg * f.league_sf_home_xgpg
    END AS home_xg,
    CASE
        WHEN t.tier = {tier} THEN t.home_xga
        ELSE t.home_xga * f.league_sf_home_xgapg
    END AS home_xga,
    CASE
        WHEN t.tier = {tier} THEN t.away_g
        ELSE t.away_g * f.league_sf_away_gpg
    END AS away_g,
    CASE
        WHEN t.tier = {tier} THEN t.away_ga
        ELSE t.away_ga * f.league_sf_away_gapg
    END AS away_ga,
    CASE
        WHEN t.tier = {tier} THEN t.away_xg
        ELSE t.away_xg * f.league_sf_away_xgpg
    END AS away_xg,
    CASE
        WHEN t.tier = {tier} THEN t.away_xga
        ELSE t.away_xga * f.league_sf_away_xgapg
    END AS away_xga,
    a.mean_home_g,
    a.mean_home_xg,
    a.mean_away_g,
    a.mean_away_xg
FROM avg_ratings a
LEFT JOIN teams t ON a.tier = t.next_tier AND a.country = t.country
LEFT JOIN factors f ON a.tier = f.next_tier AND a.country = f.country
),
calcs AS (
SELECT
    team,
    (mean_home_g * (1-{xg_factor})) + (mean_home_xg * {xg_factor}) AS smoothed_mean_home_g,
    (mean_away_g * (1-{xg_factor})) + (mean_away_xg * {xg_factor}) AS smoothed_mean_away_g,
    ((home_g * (1-{xg_factor})) + (home_xg * {xg_factor})) AS smoothed_home_g,
    ((home_ga * (1-{xg_factor})) + (home_xga * {xg_factor})) AS smoothed_home_ga,
    ((away_g * (1-{xg_factor})) + (away_xg * {xg_factor})) AS smoothed_away_g,
    ((away_ga * (1-{xg_factor})) + (away_xga * {xg_factor})) AS smoothed_away_ga
FROM joined
)
SELECT
    team,
    smoothed_home_g / smoothed_mean_home_g AS home_attack_strength,
    smoothed_home_ga / smoothed_mean_away_g AS home_defense_strength,
    smoothed_away_g / smoothed_mean_away_g AS away_attack_strength,
    smoothed_away_ga / smoothed_mean_home_g AS away_defense_strength
FROM calcs;
"""
    con = ddb.connect(database='data/football_data.db', read_only=True)
    df = con.execute(query.format(season_end_year=season_end_year, tier=tier, country=country, xg_factor=xg_factor)).fetch_df()
    con.close()
    return df

def get_points(home_goals, away_goals):
    if home_goals > away_goals:
        return 3, 0
    elif home_goals < away_goals:
        return 0, 3
    else:
        return 1, 1
    