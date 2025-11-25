import csv
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

def read_last_season_stats(season_end_year: int = 2025, country: str = 'England', tier: int = 1, xg_factor: float = 0.6):
    query = """
WITH params AS (
    SELECT
        ?::INT    AS season_end_year,
        ?::INT    AS tier,
        ?::TEXT   AS country,
        ?::DOUBLE AS xg_factor
),
avg_ratings AS (
    SELECT 
        r.country,
        r.tier,
        MEAN(r.home_g/r.home_p) as mean_home_g,
        MEAN(r.home_xg/r.home_p) as mean_home_xg,
        MEAN(r.away_g/r.away_p) as mean_away_g,
        MEAN(r.away_xg/r.away_p) as mean_away_xg
    FROM ratings r, params p 
    WHERE r.country = p.country AND r.season_end_year = p.season_end_year - 1 AND r.tier = p.tier
    GROUP BY r.country, r.tier
),
factors AS (
    SELECT 
        i.country,
        i.tier AS next_tier,
        i.prev_tier AS tier,
        i.league_sf_home_gpg,
        i.league_sf_home_xgpg,
        i.league_sf_home_gapg,
        i.league_sf_home_xgapg,
        i.league_sf_away_gpg,
        i.league_sf_away_xgpg,
        i.league_sf_away_gapg,
        i.league_sf_away_xgapg
    FROM interleague_factors i, params p
    WHERE i.country = p.country AND i.tier = p.tier AND i.prev_tier = p.tier + 1
),
teams AS (
    SELECT 
        r.country,
        r.tier,
        r.next_tier,
        r.season_end_year,
        r.team,
        r.home_g/r.home_p AS home_g, 
        r.home_ga/r.home_p AS home_ga,
        r.home_xg/r.home_p AS home_xg,
        r.home_xga/r.home_p AS home_xga,
        r.away_g/r.away_p AS away_g,
        r.away_ga/r.away_p AS away_ga,
        r.away_xg/r.away_p AS away_xg,
        r.away_xga/r.away_p AS away_xga
    FROM ratings r, params p
    WHERE r.season_end_year = p.season_end_year - 1 AND r.next_tier = p.tier AND r.country = p.country
),
joined AS (
    SELECT
    a.country,
    t.tier,
    t.next_tier,
    t.season_end_year,
    t.team,
    CASE
        WHEN t.tier = p.tier THEN t.home_g
        ELSE t.home_g * f.league_sf_home_gpg
    END AS home_g,
    CASE
        WHEN t.tier = p.tier THEN t.home_ga
        ELSE t.home_ga * f.league_sf_home_gapg
    END AS home_ga,
    CASE
        WHEN t.tier = p.tier THEN t.home_xg
        ELSE t.home_xg * f.league_sf_home_xgpg
    END AS home_xg,
    CASE
        WHEN t.tier = p.tier THEN t.home_xga
        ELSE t.home_xga * f.league_sf_home_xgapg
    END AS home_xga,
    CASE
        WHEN t.tier = p.tier THEN t.away_g
        ELSE t.away_g * f.league_sf_away_gpg
    END AS away_g,
    CASE
        WHEN t.tier = p.tier THEN t.away_ga
        ELSE t.away_ga * f.league_sf_away_gapg
    END AS away_ga,
    CASE
        WHEN t.tier = p.tier THEN t.away_xg
        ELSE t.away_xg * f.league_sf_away_xgpg
    END AS away_xg,
    CASE
        WHEN t.tier = p.tier THEN t.away_xga
        ELSE t.away_xga * f.league_sf_away_xgapg
    END AS away_xga,
    a.mean_home_g,
    a.mean_home_xg,
    a.mean_away_g,
    a.mean_away_xg
FROM avg_ratings a
LEFT JOIN teams t ON a.tier = t.next_tier AND a.country = t.country
LEFT JOIN factors f ON a.tier = f.next_tier AND a.country = f.country
JOIN params p ON TRUE
),
calcs AS (
SELECT
    j.team,
    (j.mean_home_g * (1-p.xg_factor)) + (j.mean_home_xg * p.xg_factor) AS smoothed_mean_home_g,
    (j.mean_away_g * (1-p.xg_factor)) + (j.mean_away_xg * p.xg_factor) AS smoothed_mean_away_g,
    ((j.home_g * (1-p.xg_factor)) + (j.home_xg * p.xg_factor)) AS smoothed_home_g,
    ((j.home_ga * (1-p.xg_factor)) + (j.home_xga * p.xg_factor)) AS smoothed_home_ga,
    ((j.away_g * (1-p.xg_factor)) + (j.away_xg * p.xg_factor)) AS smoothed_away_g,
    ((j.away_ga * (1-p.xg_factor)) + (j.away_xga * p.xg_factor)) AS smoothed_away_ga
FROM joined j
JOIN params p on TRUE
)
SELECT
    team,
    smoothed_home_g / smoothed_mean_home_g AS home_attack_strength,
    smoothed_home_ga / smoothed_mean_away_g AS home_defence_strength,
    smoothed_away_g / smoothed_mean_away_g AS away_attack_strength,
    smoothed_away_ga / smoothed_mean_home_g AS away_defence_strength
FROM calcs;
"""
    con = ddb.connect(database='data/football_data.db', read_only=True)
    df = con.execute(query, [season_end_year, tier, country, xg_factor]).fetch_df()
    con.close()
    return df

def get_points(home_goals, away_goals):
    if home_goals > away_goals:
        return 3, 0
    elif home_goals < away_goals:
        return 0, 3
    else:
        return 1, 1
    