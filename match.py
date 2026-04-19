import numpy as np
from utils import get_points
from scipy.stats import poisson
from team import Team
from typing import Dict, Any, List, Optional, Tuple

class Match:
    def __init__(self, teams: Dict[str, Team], fixture: Dict[str, Any], league_avg_home: float, league_avg_away: float, xG_factor: float) -> None:
        for key in ['Date', 'Home', 'Away']:
            if key not in fixture:
                raise ValueError(f"Missing key '{key}' in fixture: {fixture}")
        if fixture['Home'] not in teams or fixture['Away'] not in teams:
            raise ValueError(f"Fixture references unknown teams: {fixture['Home']} or {fixture['Away']}")
        self.date: str = fixture['Date']
        self.home_team: Team = teams[fixture['Home']]
        self.away_team: Team = teams[fixture['Away']]
        self.xG_factor: float = xG_factor
        self.match_expectation: Tuple[float, float] = self.get_match_expectation(league_avg_home, league_avg_away)

    def __repr__(self) -> str:
        return f"Match({self.home_team.name} vs {self.away_team.name} on {self.date})"

    def __str__(self) -> str:
        return f"{self.home_team.name} vs {self.away_team.name} on {self.date}"

    def get_match_expectation(self, league_avg_home: float, league_avg_away: float) -> Tuple[float, float]:
        home_expected_goals = league_avg_home * self.home_team.home_attack_strength * self.away_team.away_defence_strength
        away_expected_goals = league_avg_away * self.away_team.away_attack_strength * self.home_team.home_defence_strength
        return home_expected_goals, away_expected_goals

    @classmethod
    def from_fixtures(
        cls,
        teams: Dict[str, Team],
        fixtures: List[Dict[str, Any]],
        league_avg_home: float,
        league_avg_away: float,
        xG_factor: float,
        max_goals: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ) -> List['Match']:
        if not isinstance(fixtures, list) or not all(isinstance(f, dict) for f in fixtures):
            raise TypeError("fixtures must be a list of dicts")
        
        matches: List[Match] = []
        
        if cls.__name__ == 'MarketsMatch':
            limit = max_goals if max_goals is not None else 6
            for fixture in fixtures:
                matches.append(cls(teams, fixture, league_avg_home, league_avg_away, xG_factor, max_goals=limit))
        elif cls.__name__ == 'SimmedMatch':
            for fixture in fixtures:
                matches.append(cls(teams, fixture, league_avg_home, league_avg_away, xG_factor, rng=rng))
        else:
            for fixture in fixtures:
                matches.append(cls(teams, fixture, league_avg_home, league_avg_away, xG_factor))
        
        return matches

class MarketsMatch(Match):
    def __init__(self, teams: Dict[str, Team], fixture: Dict[str, Any], league_avg_home: float, league_avg_away: float, xG_factor: float, max_goals: int = 6) -> None:
        super().__init__(teams, fixture, league_avg_home, league_avg_away, xG_factor)
        self.max_goals: int = max_goals
        self.score_matrix: np.ndarray = self.get_score_matrix()
        self.markets: Dict[str, float] = self.get_match_markets()

    def get_score_matrix(self) -> np.ndarray:
        home_distribution = poisson.pmf(np.arange(self.max_goals + 1), self.match_expectation[0])
        away_distribution = poisson.pmf(np.arange(self.max_goals + 1), self.match_expectation[1])
        score_matrix = np.outer(home_distribution, away_distribution)

        score_matrix[-1,:-1] += (1-home_distribution.sum()) * away_distribution[:-1]
        score_matrix[:-1,-1] += home_distribution[:-1] * (1-away_distribution.sum())
        score_matrix[-1,-1] += (1-home_distribution.sum()) * (1-away_distribution.sum())

        return score_matrix

    def get_match_markets(self) -> Dict[str, float]:
        home_win = np.tril(self.score_matrix,-1).sum()
        draw = np.trace(self.score_matrix)
        away_win = np.triu(self.score_matrix, 1).sum()
        btts = 1- (self.score_matrix[0,:].sum() + self.score_matrix[:,0].sum() - self.score_matrix[0,0])
        total_goals = np.add.outer(np.arange(self.score_matrix.shape[0]), np.arange(self.score_matrix.shape[1]))
        over_2_5  = self.score_matrix[total_goals > 2.5].sum()
        under_2_5 = 1 - over_2_5
        return {
            "P(Home Win)": home_win,
            "P(Draw)": draw,
            "P(Away Win)": away_win,
            "P(BTTS)": btts,
            "P(Over 2.5 Goals)": over_2_5,
            "P(Under 2.5 Goals)": under_2_5
        }

class SimmedMatch(Match):
    def __init__(self, teams: Dict[str, Team], fixture: Dict[str, Any], league_avg_home: float, league_avg_away: float, xG_factor: float, rng: np.random.Generator) -> None:
        super().__init__(teams, fixture, league_avg_home, league_avg_away, xG_factor)
        self.sim_result: Dict[str, Any] = self.get_sim_result(rng)

    def get_sim_result(self, rng: np.random.Generator) -> Dict[str, Any]:
        home_exp, away_exp = self.match_expectation
        home_goals = rng.poisson(home_exp)
        away_goals = rng.poisson(away_exp)
        home_points, away_points = get_points(home_goals, away_goals)
        trial_result = {
            "Date": self.date,
            "Home": self.home_team.name,
            "Away": self.away_team.name,
            "HomeGoals": home_goals,
            "AwayGoals": away_goals,
            "Home_xG": home_exp,
            "Away_xG": away_exp,
            "Home_pts": home_points,
            "Away_pts": away_points
        }
        return trial_result