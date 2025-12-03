from line_profiler import profile
from typing import Dict, List, Type, Tuple, Any

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
    def teams_from_results(
        cls: Type['Team'], 
        results: List[Dict[str, Any]], 
        xG_factor: float = 0.6, 
        last_season_factor: float = None, 
        last_season_strengths: List[Dict[str, Any]] = None
    ) -> Dict[str, 'Team']:
        
        teams: Dict[str, Team] = {}
        if last_season_strengths:
            for row in last_season_strengths:
                name = row['team']
                if isinstance(name, str) and name and name not in teams:
                    teams[name] = cls(name)

        if results:
            required_keys = {'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Home_xG', 'Away_xG', 'Home_pts', 'Away_pts'}
            for row in results:
                if not required_keys.issubset(row.keys()):
                    missing = required_keys - set(row.keys())
                    raise ValueError(f"Missing columns in results {row}: {missing}")

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

        if not last_season_strengths or not isinstance(last_season_factor, (float, int)) or last_season_factor == 0:
            Team.calculate_team_strengths(teams, league_avg_home, league_avg_away, xG_factor, init=True)
        else:
            Team.calculate_team_strengths(
                teams, league_avg_home, league_avg_away, xG_factor,
                last_season_factor, last_season_strengths, init=True)
        return teams

    @staticmethod
    def calculate_team_strengths(
        teams: Dict[str, 'Team'], 
        league_avg_home: float, 
        league_avg_away: float, 
        xG_factor: float = 0.6, 
        last_season_factor: float = None,
        last_season_strengths: List[Dict[str, Any]] = None, 
        init: bool = False
        ) -> None:
        
        for team in teams.values():
            if team.home_games_played > 0 and league_avg_home > 0:
                smoothed_home_goals = (1-xG_factor)*team.home_goals + xG_factor*team.home_xg
                smoothed_home_goals_against = (1-xG_factor)*team.home_goals_against + xG_factor*team.home_xga
                team.home_attack_strength_cs = round(smoothed_home_goals / (team.home_games_played * league_avg_home), 2)
                team.home_defence_strength_cs = round(smoothed_home_goals_against / (team.home_games_played * league_avg_away), 2)

            if team.away_games_played > 0 and league_avg_away > 0:
                smoothed_away_goals = (1-xG_factor)*team.away_goals + xG_factor*team.away_xg
                smoothed_away_goals_against = (1-xG_factor)*team.away_goals_against + xG_factor*team.away_xga
                team.away_attack_strength_cs = round(smoothed_away_goals / (team.away_games_played * league_avg_away), 2)
                team.away_defence_strength_cs = round(smoothed_away_goals_against / (team.away_games_played * league_avg_home), 2)
           
        if last_season_factor is None or last_season_factor == 0:
            for team in teams.values():
                team.home_attack_strength = team.home_attack_strength_cs
                team.home_defence_strength = team.home_defence_strength_cs
                team.away_attack_strength = team.away_attack_strength_cs
                team.away_defence_strength = team.away_defence_strength_cs
        else:
            if init:
                ls_map: Dict[str, Dict[str, float]] = {}
                for row in last_season_strengths:
                    name = row['team']
                    if not isinstance(name, str):
                        continue
                    ls_map[name] = row

                for team in teams.values():
                    row = ls_map.get(team.name, {})
                    team.home_attack_strength_ls = float(row.get('home_attack_strength', 0.0))
                    team.home_defence_strength_ls = float(row.get('home_defence_strength', 0.0))
                    team.away_attack_strength_ls = float(row.get('away_attack_strength', 0.0))
                    team.away_defence_strength_ls = float(row.get('away_defence_strength', 0.0))

            for team in teams.values():
                lsf_ha = 1.0 if team.home_attack_strength_cs == 0.0 else last_season_factor
                lsf_hd = 1.0 if team.home_defence_strength_cs == 0.0 else last_season_factor
                lsf_aa = 1.0 if team.away_attack_strength_cs == 0.0 else last_season_factor
                lsf_ad = 1.0 if team.away_defence_strength_cs == 0.0 else last_season_factor
                team.home_attack_strength = Team.combine_strengths(team.home_attack_strength_cs, team.home_attack_strength_ls, lsf_ha)
                team.home_defence_strength = Team.combine_strengths(team.home_defence_strength_cs, team.home_defence_strength_ls, lsf_hd)
                team.away_attack_strength = Team.combine_strengths(team.away_attack_strength_cs, team.away_attack_strength_ls, lsf_aa)
                team.away_defence_strength = Team.combine_strengths(team.away_defence_strength_cs, team.away_defence_strength_ls, lsf_ad)

    @staticmethod
    def combine_strengths(cs_strength: float, ls_strength: float, last_season_factor: float) -> float:
        if cs_strength == 0:
            return ls_strength
        else:
            return (ls_strength * last_season_factor) + (cs_strength * (1 - last_season_factor))

    @profile
    @staticmethod
    def update_teams(
        teams: Dict[str, 'Team'], 
        new_results: list, 
        xG_factor: float = 0.6, 
        last_season_factor: float = None
        ) -> None:

        required_keys = {'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Home_xG', 'Away_xG', 'Home_pts', 'Away_pts'}
        for row in new_results:
            if not required_keys.issubset(row.keys()):
                missing = required_keys - set(row.keys())
                raise ValueError(f"Missing keys in new_results row: {missing}")
            

            home = row['Home']
            away = row['Away']
            home_goals = int(row['HomeGoals'])
            away_goals = int(row['AwayGoals'])
            home_xg = float(row['Home_xG'])
            away_xg = float(row['Away_xG'])
            home_pts = int(row['Home_pts'])
            away_pts = int(row['Away_pts'])

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
        Team.calculate_team_strengths(teams, league_avg_home, league_avg_away, xG_factor, last_season_factor, init = False)


    @staticmethod
    def get_league_averages(teams: Dict[str, 'Team'], xG_factor: float = 0.6) -> Tuple[float, float]:
        home_goals = sum(team.home_goals for team in teams.values())
        away_goals = sum(team.away_goals for team in teams.values())
        home_xg = sum(team.home_xg for team in teams.values())
        away_xg = sum(team.away_xg for team in teams.values())
        games_count = sum(team.home_games_played for team in teams.values())
        if games_count <= 0:
            return 1.0, 1.0 #Default when no games played
        
        smooth_home_goals = (1-xG_factor)*home_goals + xG_factor*home_xg
        smooth_away_goals = (1-xG_factor)*away_goals + xG_factor*away_xg
        league_avg_home = smooth_home_goals / games_count
        league_avg_away = smooth_away_goals / games_count
        return league_avg_home, league_avg_away