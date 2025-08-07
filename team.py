from datetime import datetime
import pandas as pd
class Team:
    def __init__(self, name):
        self.name = name
        self.home_games_played = 0
        self.away_games_played = 0
        self.home_points = 0
        self.away_points = 0
        self.home_goals = 0
        self.away_goals = 0
        self.home_goals_against = 0
        self.away_goals_against = 0
        self.home_xg = 0.0
        self.away_xg = 0.0
        self.home_xga = 0.0
        self.away_xga = 0.0
        self.home_attack_strength = 0.0
        self.away_attack_strength = 0.0
        self.home_defence_strength = 0.0
        self.away_defence_strength = 0.0


    def __repr__(self):
        return (f"Team({self.name}, HGP={self.home_games_played}, AGP={self.away_games_played}, "
                f"HG={self.home_goals}, AG={self.away_goals}, "
                f"HGA={self.home_goals_against}, AGA={self.away_goals_against}, "
                f"HxG={self.home_xg}, AxG={self.away_xg}, "
                f"HxGA={self.home_xga}, AxGA={self.away_xga}, "
                f"HAS={self.home_attack_strength}, AAS={self.away_attack_strength}, "
                f"HDS={self.home_defence_strength}, ADS={self.away_defence_strength}, "
                f"HP={self.home_points}, AP={self.away_points})")
   
    @classmethod
    def teams_from_results(cls, results, xG_factor=0.6):
        teams = {}
        league_avg_home, league_avg_away = Team.get_league_averages(results, xG_factor)

        for _,row in results.iterrows():
            home = row['Home']
            away = row['Away']
            home_goals = int(row['HomeGoals'])
            away_goals = int(row['AwayGoals'])
            home_xg = float(row['Home_xG'])
            away_xg = float(row['Away_xG'])
            home_pts = int(row['Home_pts'])
            away_pts = int(row['Away_pts'])

            if home not in teams:
                teams[home] = cls(home)
            if away not in teams:
                teams[away] = cls(away)

            # Update games played
            teams[home].home_games_played += 1
            teams[away].away_games_played += 1

            # Update goals
            teams[home].home_goals += home_goals
            teams[away].away_goals += away_goals
            teams[home].home_goals_against += away_goals
            teams[away].away_goals_against += home_goals
            
            # Update xG 
            teams[home].home_xg += home_xg
            teams[away].away_xg += away_xg
            teams[home].home_xga += away_xg
            teams[away].away_xga += home_xg

            # Update Points scored
            teams[home].home_points += home_pts
            teams[away].away_points += away_pts

        # Calculate team strengths
        Team.calculate_team_strengths(teams,league_avg_home, league_avg_away, xG_factor)
        return teams
    
    @staticmethod
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
    
    @staticmethod
    def calculate_team_strengths(teams,league_avg_home, league_avg_away, xG_factor=0.6): 
        for team in teams.values():
            smoothed_home_goals = (1-xG_factor)*team.home_goals + xG_factor*team.home_xg
            smoothed_home_goals_against = (1-xG_factor)*team.home_goals_against + xG_factor*team.home_xga
            team.home_attack_strength = round(smoothed_home_goals / (team.home_games_played * league_avg_home), 2)
            team.home_defence_strength = round(smoothed_home_goals_against / (team.home_games_played * league_avg_away), 2)

            smoothed_away_goals = (1-xG_factor)*team.away_goals + xG_factor*team.away_xg
            smoothed_away_goals_against = (1-xG_factor)*team.away_goals_against + xG_factor*team.away_xga
            team.away_attack_strength = round(smoothed_away_goals / (team.away_games_played * league_avg_away), 2)
            team.away_defence_strength = round(smoothed_away_goals_against / (team.away_games_played * league_avg_home), 2)
