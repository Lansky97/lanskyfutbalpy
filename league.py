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
        self.total_home_goals: int = 0
        self.total_away_goals: int = 0
        self.total_home_xg: float = 0.0
        self.total_away_xg: float = 0.0
        self.games_played: int = len(self.results)
        self.league_avg_home: float = 0.0
        self.league_avg_away: float = 0.0

        for result in self.results:
            self.total_home_goals += int(result['HomeGoals'])
            self.total_away_goals += int(result['AwayGoals'])
            self.total_home_xg += float(result['Home_xG'])
            self.total_away_xg += float(result['Away_xG'])  

        xG_inverse = 1 - xG_factor

        if self.games_played == 0:
            self.league_avg_home, self.league_avg_away = 1.0, 1.0
        else:
            smooth_home_goals = xG_inverse*self.total_home_goals + xG_factor*self.total_home_xg
            smooth_away_goals = xG_inverse*self.total_away_goals + xG_factor*self.total_away_xg
            self.league_avg_home = smooth_home_goals / self.games_played
            self.league_avg_away = smooth_away_goals / self.games_played


    def __repr__(self) -> str:
        return f"League({self.name}, Teams: {len(self.teams)}, Played: {self.games_played}, Fixtures: {len(self.fixtures)})"
       
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
        lge.teams = Team.teams_from_results(
            lge.results, lge.league_avg_home, lge.league_avg_away, lge.xG_factor
            )
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

        last_season_stats = read_last_season_stats(season_end_year=season_end_year, country=country, tier=tier, xg_factor=xG_factor)
        
        lge = cls(league_name, matches, date_cutoff, xG_factor, last_season_factor)
        lge.teams = Team.teams_from_results(
            lge.results, lge.league_avg_home, lge.league_avg_away,
            lge.xG_factor, lge.last_season_factor, last_season_stats)
        lge.last_season_stats = last_season_stats
        lge.league_table = lge.generate_league_table()
        return lge
    
    def update_league(self, new_results: List[Dict[str, Any]]) -> None:
        for result in new_results:
            self.total_home_goals += int(result['HomeGoals'])
            self.total_away_goals += int(result['AwayGoals'])
            self.total_home_xg += float(result['Home_xG'])
            self.total_away_xg += float(result['Away_xG'])
        
        self.games_played += len(new_results)
        xG_inverse = 1 - self.xG_factor
        smooth_home_goals = xG_inverse*self.total_home_goals + self.xG_factor*self.total_home_xg
        smooth_away_goals = xG_inverse*self.total_away_goals + self.xG_factor*self.total_away_xg
        self.league_avg_home = smooth_home_goals / self.games_played
        self.league_avg_away = smooth_away_goals / self.games_played
        self.results.extend(new_results)
        
        Team.update_teams(
            self.teams, new_results, self.league_avg_home, self.league_avg_away,
            self.xG_factor, self.last_season_factor
            )

    def generate_fixtures(self) -> List[Dict[str, Any]]:
        fixtures: List[Dict[str, Any]] = []
        seen_fixtures = set()
        for row in self.matches:
            if row['Date'] <= self.date_cutoff_str:
                continue

            unique_key = (row['Date'], row['Home'], row['Away'])
            if unique_key not in seen_fixtures:
                seen_fixtures.add(unique_key)
                fixtures.append({
                    'Date': row['Date'], 
                    'Home': row['Home'],
                    'Away': row['Away']
                })

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
            home_pts, away_pts = get_points(home_goals, away_goals)
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

