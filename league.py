from team import Team
from utils import read_results
from utils import get_points
from datetime import datetime
import pandas as pd

class League:
    def __init__(self, name, matches, date_cutoff, xG_factor=0.6):
        self.name = name
        self.xG_factor = xG_factor
        self.matches = matches 
        self.date_cutoff = datetime.strptime(date_cutoff,"%Y-%m-%d")
        self.fixtures = self.generate_fixtures()
        self.results = self.generate_results()
        self.teams = Team.teams_from_results(self.results, xG_factor)
        self.league_table = self.generate_league_table()

    def __repr__(self):
        return f"League({self.name}, Teams: {len(self.teams)},Played: {len(self.results)}, Fixtures: {len(self.fixtures)})"
       
    @classmethod
    def from_matches(cls, match_data = 'data/result_24_25.csv', date_cutoff = '2024-12-01', xG_factor=0.6):
        # Get league name from the first match (assuming all matches are from the same competition)
        matches = read_results(match_data)
        league_name = matches[0]['Competition_Name'] if matches else "Unknown League"
        return cls(league_name, matches, date_cutoff, xG_factor)
        
    def generate_fixtures(self):
        fixtures = []
        for row in self.matches:
            match_date = datetime.strptime(row['Date'], "%Y-%m-%d")
            if match_date <= self.date_cutoff:
                continue

            fixture = {'Date': row['Date'],
                       'Home': row['Home'], 
                       'Away': row['Away']}
            if fixture not in fixtures:
                fixtures.append(fixture)
        fixtures.sort(key=lambda x: x['Date'])
        fixtures = pd.DataFrame(fixtures)
        return fixtures
    
    def update_fixtures(self, update_date):
        self.fixtures = self.fixtures[self.fixtures['Date'] > update_date]
    
    def generate_results(self):
        results = []
        for row in self.matches:
            match_date = datetime.strptime(row['Date'], "%Y-%m-%d")
            if match_date > self.date_cutoff:
                continue
            
            results.append({
                'Date': row['Date'],
                'Home': row['Home'],
                'Away': row['Away'],
                'HomeGoals': row['HomeGoals'],
                'AwayGoals': row['AwayGoals'],
                'Home_xG': row['Home_xG'],
                'Away_xG': row['Away_xG'],
                'Home_pts': get_points(int(row['HomeGoals']), int(row['AwayGoals']))[0],
                'Away_pts': get_points(int(row['HomeGoals']), int(row['AwayGoals']))[1]
                })
            
        results.sort(key=lambda x: datetime.strptime(x['Date'], "%Y-%m-%d"))
        results = pd.DataFrame(results)
        return results
    
    def update_results(self, new_results):
        new_results = pd.DataFrame(new_results)
        new_results = new_results.reindex(columns=self.results.columns)
        self.results = pd.concat([self.results, new_results], ignore_index=True)
    
    def generate_league_table(self):
        table = []
        for team in self.teams.values():
            played = team.home_games_played + team.away_games_played
            total_points = team.home_points + team.away_points
            total_goals = team.home_goals + team.away_goals
            total_goals_against = team.home_goals_against + team.away_goals_against
            total_xg = team.home_xg + team.away_xg
            total_xga = team.home_xga + team.away_xga
            goal_diff = total_goals - total_goals_against

            table.append({
                'Team': team.name,
                'Played': played,
                'Points': total_points,
                'Goals': total_goals,
                'Goals Against': total_goals_against,
                'Goal Difference': goal_diff,
                'xG': total_xg,
                'xGA': total_xga
                })
            
        table.sort(key=lambda x: (x['Points'], x['Goal Difference'], x['Goals']), reverse=True)
        table = pd.DataFrame(table)
        table.insert(0, 'Pos', range(1, len(table) + 1))
        return table

