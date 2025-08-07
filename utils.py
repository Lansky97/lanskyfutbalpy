import csv
import pandas as pd

def read_results(file_path):
    results = []
    with open(file_path,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            results.append(row)
    return results

def get_league_averages(results, xG_factor=0.6):
        home_goals = pd.to_numeric(results['HomeGoals'], errors='coerce')
        away_goals = pd.to_numeric(results['AwayGoals'], errors='coerce')
        home_xg = pd.to_numeric(results['Home_xG'], errors='coerce')
        away_xg = pd.to_numeric(results['Away_xG'], errors='coerce')

        total_home_goals = home_goals.sum()
        total_away_goals = away_goals.sum()
        total_home_xg = home_xg.sum()
        total_away_xg = away_xg.sum()

        league_avg_home = ((1-xG_factor)*total_home_goals + xG_factor*total_home_xg) / len(results)
        league_avg_away = ((1-xG_factor)*total_away_goals + xG_factor*total_away_xg)/ len(results)

        return league_avg_home, league_avg_away

def get_points(home_goals, away_goals):
    if home_goals > away_goals:
        return 3, 0
    elif home_goals < away_goals:
        return 0, 3
    else:
        return 1, 1
    