import csv
from itertools import product
import pandas as pd
import numpy as np
from datetime import datetime

matches = pd.read_csv('/Users/lansky/Documents/Python/football_simulation/data/result_24_25.csv')

def matches_to_team_ratings(matches, date = None, xG_factor = 0.6):
    if date is None:
        date = datetime.today().date()
    else:
        if isinstance(date, str):
            data = pd.to_datetime(date).date()
        elif isinstance(date,pd.Timestamp):
            date = date.date()

    matches['Date'] = pd.to_datetime(matches['Date']).dt.date

    matches_filtered = matches[(matches['HomeGoals'].notnull()) & (matches['Date'] < date)].copy()

    matches_filtered = matches_filtered[['Home', 'HomeGoals', 'Home_xG', 'Away', 'AwayGoals', 'Away_xG']].rename(columns={'Home': 'home_team', 'HomeGoals': 'home_goals', 'Home_xG': 'home_xG', 'Away': 'away_team', 'AwayGoals': 'away_goals', 'Away_xG': 'away_xG'})

    matches_filtered.reset_index(drop=True, inplace=True)
    matches_filtered['gameId'] = matches_filtered.index + 1

    home_df = matches_filtered[['gameId', 'home_team', 'home_goals', 'home_xG']].copy()
    home_df['venue'] = 'home'
    home_df.rename(columns={'home_team': 'team', 'home_goals': 'goals', 'home_xG': 'xG' }, inplace=True)

    away_df = matches_filtered[['gameId', 'away_team', 'away_goals', 'away_xG']].copy()
    away_df['venue'] = 'away'
    away_df.rename(columns={'away_team': 'team', 'away_goals': 'goals', 'away_xG': 'xG'}, inplace=True)

    matches_long = pd.concat([home_df, away_df], ignore_index=True)

    matches_long['xG'] = matches_long['xG'].fillna(matches_long['goals'])
    matches_long['gA'] = matches_long.groupby('gameId')['goals'].transform('sum') - matches_long['goals']
    matches_long['xGA'] = matches_long.groupby('gameId')['xG'].transform('sum') - matches_long['xG']

    matches_long['smoothed_goals'] = ((1 - xG_factor) * matches_long['goals'] + xG_factor * matches_long['xG'])
    matches_long['smoothed_goalsA'] = ((1 - xG_factor) * matches_long['gA'] + xG_factor * matches_long['xGA'])

    group_cols = ['venue', 'team']
    matches_long['expG'] = matches_long.groupby(group_cols)['smoothed_goals'].transform('mean').round(2)
    matches_long['expGA'] = matches_long.groupby(group_cols)['smoothed_goalsA'].transform('mean').round(2)
    matches_long['Total_smoothed_goals'] = matches_long.groupby(group_cols)['smoothed_goals'].transform('sum')
    matches_long['Total_smoothed_goalsA'] = matches_long.groupby(group_cols)['smoothed_goalsA'].transform('sum')
    matches_long['games_played'] = matches_long.groupby(group_cols)['smoothed_goals'].transform('count')

    matches_long = matches_long.drop_duplicates(subset=group_cols)

    league_expG = matches_long.groupby('venue')['expG'].transform('mean')
    matches_long['league_expG'] = league_expG

    matches_long.sort_values(['team', 'venue'], inplace=True)

    teams_list = matches_filtered['home_team'].unique()
    venues = ['home', 'away']
    grid = pd.DataFrame(list(product(teams_list, venues)), columns=['team', 'venue'])

    final_df = grid.merge(matches_long, on=['team', 'venue'], how='left')

    for col in ['games_played', 'Total_smoothed_goals', 'Total_smoothed_goalsA']:
            final_df[col] = np.where(final_df['expG'].isna(), 0, final_df[col])

    for venue in final_df['venue'].unique():
            venue_mask = final_df['venue'] == venue
            for col in ['expG', 'expGA', 'league_expG']:
                mean_value = final_df.loc[venue_mask, col].mean()
                final_df.loc[venue_mask & final_df[col].isna(), col] = mean_value

    final_df.reset_index(drop=True, inplace=True)
    return final_df


team_ratings = matches_to_team_ratings(matches)
print(team_ratings)