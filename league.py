from line_profiler import profile
from team import Team
from utils import read_schedule
from utils import get_points
from utils import read_last_season_stats
from datetime import datetime

from typing import List, Dict, Any, Optional

class League:
    def __init__(self, name: str, matches: List[Dict[str, Any]], date_cutoff: str, xG_factor: float = 0.6, last_season_factor: float = None) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not isinstance(matches, list) or not all(isinstance(m, dict) for m in matches):
            raise TypeError("matches must be a list of dicts.")
        try:
            self.date_cutoff = datetime.strptime(date_cutoff, "%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"date_cutoff must be a string in YYYY-MM-DD format: {date_cutoff}") from e
        self.date_cutoff_str = date_cutoff
        if not isinstance(xG_factor, float):
            raise TypeError("xG_factor must be a float.")
        self.name: str = name
        self.xG_factor: float = xG_factor
        self.last_season_factor: float = last_season_factor
        self.matches: List[Dict[str, Any]] = matches
        self.fixtures: List[Dict[str, Any]] = self.generate_fixtures()
        self.results: List[Dict[str, Any]] = self.generate_results()            
        self.teams: Dict[str, Team] = {}
        self.league_table: List[Dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"League({self.name}, Teams: {len(self.teams)}, Played: {len(self.results)}, Fixtures: {len(self.fixtures)})"
       
    @classmethod
    def from_matches(
        cls,
        match_data: str = 'data/result_24_25.csv',
        date_cutoff: str = '2024-12-01',
        xG_factor: float = 0.6
    ) -> 'League':
        matches = read_schedule(filepath = match_data)
        if not matches or 'Competition_Name' not in matches[0]:
            league_name = "Unknown League"
        else:
            league_name = matches[0]['Competition_Name']
        lge = cls(league_name, matches, date_cutoff, xG_factor)
        lge.teams = Team.teams_from_results(lge.results,lge.xG_factor)
        lge.league_table = lge.generate_league_table()
        return lge
    
    @classmethod
    def from_database(
        cls,
        season_end_year: int = 2025,
        league: str = 'Premier_League',
        tier: int = 1,
        date_cutoff: str = '2024-12-01',
        xG_factor: float = 0.6,
        last_season_factor: float = 0.5
    ) -> 'League':
        matches = read_schedule(season_end_year=season_end_year, league=league)
        if not matches or 'Competition_Name' not in matches[0]:
            league_name = "Unknown League"
            country = "England" # Default fallback
        else:
            league_name = matches[0]['Competition_Name']
            country = matches[0].get('Country', 'England')

        if last_season_factor == 0.0 or last_season_factor is None:
            last_season_stats = None
        else:    
            last_season_stats = read_last_season_stats(season_end_year=season_end_year, country=country, tier=tier, xg_factor=xG_factor)
        
        lge = cls(league_name, matches, date_cutoff, xG_factor, last_season_factor)
        lge.teams = Team.teams_from_results(lge.results, lge.xG_factor, lge.last_season_factor, last_season_stats)
        lge.last_season_stats = last_season_stats
        lge.league_table = lge.generate_league_table()
        return lge
    
    def update_league(self, new_results: List[Dict[str, Any]]) -> None:
        if not isinstance(new_results, list):
            raise TypeError("new_results must be a list of dicts.")
        self.update_results(new_results)
        Team.update_teams(self.teams, new_results, self.xG_factor, self.last_season_factor)

    def generate_fixtures(self) -> List[Dict[str, Any]]:
        fixtures: List[Dict[str, Any]] = []
        for row in self.matches:
            if row['Date'] <= self.date_cutoff_str:
                continue
            for key in ['Home', 'Away', 'Date']:
                if key not in row:
                    raise ValueError(f"Missing key '{key}' in fixture row: {row}")
            fixture = {'Date': row['Date'], 'Home': row['Home'], 'Away': row['Away']}
            if not any(f['Date'] == fixture['Date'] and f['Home'] == fixture['Home'] and f['Away'] == fixture['Away'] for f in fixtures):
                fixtures.append(fixture)
        fixtures.sort(key=lambda x: x['Date'])
        return fixtures
    
    def update_fixtures(self, update_date: str) -> None:
        try:
            update_dt = datetime.strptime(update_date, "%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"update_date must be a string in YYYY-MM-DD format: {update_date}") from e
        self.fixtures = [f for f in self.fixtures if f['Date'] > update_dt]
    
    def generate_results(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for row in self.matches:
            if row['Date'] > self.date_cutoff_str:
                continue
            required_keys = ['Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Home_xG', 'Away_xG']
            for key in required_keys:
                if key not in row:
                    raise ValueError(f"Missing key '{key}' in result row: {row}")
            home_goals = int(row['HomeGoals']); away_goals = int(row['AwayGoals'])
            home_xg = float(row['Home_xG']); away_xg = float(row['Away_xG'])
            home_pts, away_pts = get_points(int(row['HomeGoals']), int(row['AwayGoals']))
            results.append({
                'Date': row['Date'],
                'Home': row['Home'],
                'Away': row['Away'],
                'HomeGoals': home_goals,
                'AwayGoals': away_goals,
                'Home_xG': home_xg,
                'Away_xG': away_xg,
                'Home_pts': home_pts,
                'Away_pts': away_pts
            })
        results.sort(key=lambda x: x['Date'])
        return results
    
    @profile    
    def update_results(self, new_results: List[Dict[str, Any]]) -> None:
        if not isinstance(new_results, list):
            raise TypeError("new_results must be a list of dicts.")
        required = set(self.results[0].keys()) if self.results else {
            'Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Home_xG', 'Away_xG', 'Home_pts', 'Away_pts'
        }
        cleaned: list[Dict[str, Any]] = []
        for row in new_results: 
            if not required.issubset(row.keys()):
                missing = required - set(row.keys())
                raise ValueError(f"Missing keys in new_results row: {row}. Required keys: {missing}")
            cleaned.append({col: row[col] for col in required})
        self.results.extend(cleaned)
        self.results.sort(key=lambda x: x['Date'])
    
    def generate_league_table(self) -> List[Dict[str, Any]]:
        table: List[Dict[str, Any]] = []
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
        for i, row in enumerate(table, start=1):
            row['Pos'] = i
        return table

