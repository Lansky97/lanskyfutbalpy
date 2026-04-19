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
        league_avg_home: float,
        league_avg_away: float,
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
            for row in results:
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

        if not last_season_strengths or not isinstance(last_season_factor, (float, int)):
            Team.calculate_team_strengths(
                teams, league_avg_home, 
                league_avg_away, xG_factor, init=True)
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
        xG_inverse = 1 - xG_factor
        use_lsf = last_season_factor is not None
        if use_lsf and not 0.0 <= last_season_factor <= 1.0:
            raise ValueError("last_season_factor must be between 0.0 and 1.0 inclusive")
        inverse_lsf = (1.0 - last_season_factor) if use_lsf else 0.0

        ls_map = None
        if init and last_season_strengths:
            ls_map = {row['team']: row for row in last_season_strengths}

        for team in teams.values():

            if ls_map:
                row = ls_map.get(team.name, {})
                team.home_attack_strength_ls = float(row.get('home_attack_strength', 0.0))
                team.home_defence_strength_ls = float(row.get('home_defence_strength', 0.0))
                team.away_attack_strength_ls = float(row.get('away_attack_strength', 0.0))
                team.away_defence_strength_ls = float(row.get('away_defence_strength', 0.0))

            if team.home_games_played > 0 and league_avg_home > 0:
                smoothed_home_goals = xG_inverse*team.home_goals + xG_factor*team.home_xg
                smoothed_home_goals_against = xG_inverse*team.home_goals_against + xG_factor*team.home_xga
                team.home_attack_strength_cs = smoothed_home_goals / (team.home_games_played * league_avg_home)
                team.home_defence_strength_cs = smoothed_home_goals_against / (team.home_games_played * league_avg_away)
            else:
                if use_lsf:
                    team.home_attack_strength_cs = team.home_attack_strength_ls
                    team.home_defence_strength_cs = team.home_defence_strength_ls
                else:
                    team.home_attack_strength_cs = 1.0
                    team.home_defence_strength_cs = 1.0

            if team.away_games_played > 0 and league_avg_away > 0:
                smoothed_away_goals = xG_inverse*team.away_goals + xG_factor*team.away_xg
                smoothed_away_goals_against = xG_inverse*team.away_goals_against + xG_factor*team.away_xga
                team.away_attack_strength_cs = smoothed_away_goals / (team.away_games_played * league_avg_away) 
                team.away_defence_strength_cs = smoothed_away_goals_against / (team.away_games_played * league_avg_home)
            else:
                if use_lsf:
                    team.away_attack_strength_cs = team.away_attack_strength_ls
                    team.away_defence_strength_cs = team.away_defence_strength_ls
                else:
                    team.away_attack_strength_cs = 1.0
                    team.away_defence_strength_cs = 1.0

            if not use_lsf:
                team.home_attack_strength = team.home_attack_strength_cs
                team.home_defence_strength = team.home_defence_strength_cs
                team.away_attack_strength = team.away_attack_strength_cs
                team.away_defence_strength = team.away_defence_strength_cs
            else:
                home_attack_cs = team.home_attack_strength_cs
                if home_attack_cs == 0.0:
                    team.home_attack_strength = team.home_attack_strength_ls
                else:
                    team.home_attack_strength = (home_attack_cs * inverse_lsf) + (team.home_attack_strength_ls * last_season_factor)

                home_defence_cs = team.home_defence_strength_cs
                if home_defence_cs == 0.0:
                    team.home_defence_strength = team.home_defence_strength_ls
                else:
                    team.home_defence_strength = (home_defence_cs * inverse_lsf) + (team.home_defence_strength_ls * last_season_factor)

                away_attack_cs = team.away_attack_strength_cs
                if away_attack_cs == 0.0:
                    team.away_attack_strength = team.away_attack_strength_ls
                else:
                    team.away_attack_strength = (away_attack_cs * inverse_lsf) + (team.away_attack_strength_ls * last_season_factor)

                away_defence_cs = team.away_defence_strength_cs
                if away_defence_cs == 0.0:
                    team.away_defence_strength = team.away_defence_strength_ls
                else:
                    team.away_defence_strength = (away_defence_cs * inverse_lsf) + (team.away_defence_strength_ls * last_season_factor)

    @staticmethod
    def update_teams(
        teams: Dict[str, 'Team'], 
        new_results: list, 
        league_avg_home: float,
        league_avg_away: float,
        xG_factor: float = 0.6, 
        last_season_factor: float = None,
        ) -> None:

        for row in new_results:

            home = row['Home']
            away = row['Away']
            home_goals = int(row['HomeGoals'])
            away_goals = int(row['AwayGoals'])
            home_xg = float(row['Home_xG'])
            away_xg = float(row['Away_xG'])
            home_pts = int(row['Home_pts'])
            away_pts = int(row['Away_pts'])

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

        Team.calculate_team_strengths(
            teams, league_avg_home, league_avg_away,
            xG_factor, last_season_factor, init=False)