from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional, Type, Tuple
from utils import read_last_season_stats

class Team:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.home_games_played: int = 0
        self.away_games_played: int = 0
        self.home_points: int = 0
        self.away_points: int = 0
        self.home_goals: int = 0
        self.away_goals: int = 0
        self.home_goals_against: int = 0
        self.away_goals_against: int = 0
        self.home_xg: float = 0.0
        self.away_xg: float = 0.0
        self.home_xga: float = 0.0
        self.away_xga: float = 0.0
        self.home_attack_strength_cs: float = 0.0
        self.away_attack_strength_cs: float = 0.0
        self.home_defence_strength_cs: float = 0.0
        self.away_defence_strength_cs: float = 0.0
        self.home_attack_strength_ls: float = 0.0
        self.away_attack_strength_ls: float = 0.0
        self.home_defence_strength_ls: float = 0.0
        self.away_defence_strength_ls: float = 0.0
        self.home_attack_strength: float = 0.0
        self.away_attack_strength: float = 0.0
        self.home_defence_strength: float = 0.0
        self.away_defence_strength: float = 0.0

    def __repr__(self) -> str:
        return (f"Team({self.name}, HGP={self.home_games_played}, AGP={self.away_games_played}, "
                f"HG={self.home_goals}, AG={self.away_goals}, "
                f"HGA={self.home_goals_against}, AGA={self.away_goals_against}, "
                f"HxG={self.home_xg}, AxG={self.away_xg}, "
                f"HxGA={self.home_xga}, AxGA={self.away_xga}, "
                f"HAS={self.home_attack_strength}, AAS={self.away_attack_strength}, "
                f"HDS={self.home_defence_strength}, ADS={self.away_defence_strength}, "
                f"HP={self.home_points}, AP={self.away_points})")

    def __str__(self) -> str:
        return f"{self.name}: Points={self.home_points + self.away_points}, Goals={self.home_goals + self.away_goals}"
   
    @classmethod
    def teams_from_results(cls: Type['Team'], results: pd.DataFrame, xG_factor: float = 0.6) -> Dict[str, 'Team']:
        required_columns = {'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Home_xG', 'Away_xG', 'Home_pts', 'Away_pts'}
        if not required_columns.issubset(results.columns):
            missing = required_columns - set(results.columns)
            raise ValueError(f"Missing columns in results DataFrame: {missing}")
        teams: Dict[str, Team] = {}
        for _, row in results.iterrows():
            try:
                home = row['Home']
                away = row['Away']
                home_goals = int(row['HomeGoals'])
                away_goals = int(row['AwayGoals'])
                home_xg = float(row['Home_xG'])
                away_xg = float(row['Away_xG'])
                home_pts = int(row['Home_pts'])
                away_pts = int(row['Away_pts'])
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid row data: {row}") from e

            if home not in teams:
                teams[home] = cls(home)
            if away not in teams:
                teams[away] = cls(away)

            teams[home].home_games_played += 1
            teams[away].away_games_played += 1
            teams[home].home_goals += home_goals
            teams[away].away_goals += away_goals
            teams[home].home_goals_against += away_goals
            teams[away].away_goals_against += home_goals
            teams[home].home_xg += home_xg
            teams[away].away_xg += away_xg
            teams[home].home_xga += away_xg
            teams[away].away_xga += home_xg
            teams[home].home_points += home_pts
            teams[away].away_points += away_pts

        league_avg_home, league_avg_away = Team.get_league_averages(teams, xG_factor)
        Team.calculate_team_strengths_cs(teams, league_avg_home, league_avg_away, xG_factor)
        return teams
    
    @classmethod
    def teams_from_results_advanced(cls: Type['Team'], results: pd.DataFrame, last_season_strengths: pd.DataFrame, last_season_factor: float) -> Dict[str, 'Team']:
        teams = cls.teams_from_results(results)

        for _, row in last_season_strengths.iterrows():
            team_name = row['team']
            if team_name in teams:
                teams[team_name].home_attack_strength_ls = row.get('home_attack_strength', 0.0)
                teams[team_name].home_defence_strength_ls = row.get('home_defense_strength', 0.0)
                teams[team_name].away_attack_strength_ls = row.get('away_attack_strength', 0.0)
                teams[team_name].away_defence_strength_ls = row.get('away_defense_strength', 0.0)

        Team.calculate_team_strengths(teams, last_season_factor)

    @staticmethod
    def calculate_team_strengths(teams: Dict[str, 'Team'], last_season_factor: float = 0.5) -> None:
        for team in teams.values():
            if team.home_attack_strength_cs == 0.0:
                team.home_attack_strength = team.home_attack_strength_ls
            else:
                team.home_attack_strength = ((team.home_attack_strength_ls * last_season_factor) +
                                              (team.home_attack_strength_cs * (1 - last_season_factor)))
                
            if team.home_defence_strength_cs == 0.0:
                team.home_defence_strength = team.home_defence_strength_ls
            else:
                team.home_defence_strength = ((team.home_defence_strength_ls * last_season_factor) +
                                               (team.home_defence_strength_cs * (1 - last_season_factor)))
            if team.away_attack_strength_cs == 0.0:
                team.away_attack_strength = team.away_attack_strength_ls
            else:
                team.away_attack_strength = ((team.away_attack_strength_ls * last_season_factor) +
                                              (team.away_attack_strength_cs * (1 - last_season_factor)))
            if team.away_defence_strength_cs == 0.0:
                team.away_defence_strength = team.away_defence_strength_ls
            else:
                team.away_defence_strength = ((team.away_defence_strength_ls * last_season_factor) +
                                               (team.away_defence_strength_cs * (1 - last_season_factor)))


    @staticmethod
    def update_teams(teams: Dict[str, 'Team'], new_results: list, xG_factor: float = 0.6) -> None:
        required_keys = {'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Home_xG', 'Away_xG', 'Home_pts', 'Away_pts'}
        new_results = pd.DataFrame(new_results)
        for _,row in new_results.iterrows():
            if not required_keys.issubset(row.keys()):
                missing = required_keys - set(row.keys())
                raise ValueError(f"Missing keys in new_results row: {missing}")
            try:
                home = row['Home']
                away = row['Away']
                home_goals = int(row['HomeGoals'])
                away_goals = int(row['AwayGoals'])
                home_xg = float(row['Home_xG'])
                away_xg = float(row['Away_xG'])
                home_pts = int(row['Home_pts'])
                away_pts = int(row['Away_pts'])
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid row data: {row}") from e

            if home not in teams or away not in teams:
                raise ValueError(f"Teams {home} or {away} not found in the league.")

            teams[home].home_games_played += 1
            teams[away].away_games_played += 1
            teams[home].home_goals += home_goals
            teams[away].away_goals += away_goals
            teams[home].home_goals_against += away_goals
            teams[away].away_goals_against += home_goals
            teams[home].home_xg += home_xg
            teams[away].away_xg += away_xg
            teams[home].home_xga += away_xg
            teams[away].away_xga += home_xg
            teams[home].home_points += home_pts
            teams[away].away_points += away_pts

        league_avg_home, league_avg_away = Team.get_league_averages(teams, xG_factor)
        Team.calculate_team_strengths(teams, league_avg_home, league_avg_away, xG_factor)

    @staticmethod
    def get_league_averages(teams: Dict[str, 'Team'], xG_factor: float = 0.6) -> Tuple[float, float]:
        home_goals = sum(team.home_goals for team in teams.values())
        away_goals = sum(team.away_goals for team in teams.values())
        home_xg = sum(team.home_xg for team in teams.values())
        away_xg = sum(team.away_xg for team in teams.values())
        games_count = sum(team.home_games_played for team in teams.values())
        if games_count <= 0:
            raise ValueError("No games played in league for average calculation.")
        smooth_home_goals = (1-xG_factor)*home_goals + xG_factor*home_xg
        smooth_away_goals = (1-xG_factor)*away_goals + xG_factor*away_xg
        league_avg_home = smooth_home_goals / games_count
        league_avg_away = smooth_away_goals / games_count
        return league_avg_home, league_avg_away
    
    @staticmethod
    def calculate_team_strengths_cs(
        teams: Dict[str, 'Team'], league_avg_home: float, league_avg_away: float, xG_factor: float = 0.6
    ) -> None:
        for team in teams.values():
            if team.home_games_played > 0 and league_avg_home > 0:
                smoothed_home_goals = (1-xG_factor)*team.home_goals + xG_factor*team.home_xg
                smoothed_home_goals_against = (1-xG_factor)*team.home_goals_against + xG_factor*team.home_xga
                team.home_attack_strength_cs = round(smoothed_home_goals / (team.home_games_played * league_avg_home), 2)
                team.home_defence_strength_cs = round(smoothed_home_goals_against / (team.home_games_played * league_avg_away), 2)
            else:
                team.home_attack_strength_cs = 0.0
                team.home_defence_strength_cs = 0.0
            if team.away_games_played > 0 and league_avg_away > 0:
                smoothed_away_goals = (1-xG_factor)*team.away_goals + xG_factor*team.away_xg
                smoothed_away_goals_against = (1-xG_factor)*team.away_goals_against + xG_factor*team.away_xga
                team.away_attack_strength_cs = round(smoothed_away_goals / (team.away_games_played * league_avg_away), 2)
                team.away_defence_strength_cs = round(smoothed_away_goals_against / (team.away_games_played * league_avg_home), 2)
            else:
                team.away_attack_strength_cs = 0.0
                team.away_defence_strength_cs = 0.0