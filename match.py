import numpy as np
from utils import get_points
from scipy.stats import poisson
from team import Team

class Match:
    def __init__(self, teams, fixture, xG_factor=0.6):
        self.date = fixture['Date']
        self.home_team = teams[fixture['Home']]
        self.away_team = teams[fixture['Away']]
        self.xG_factor = xG_factor
        self.match_expectation = self.get_match_expectation(teams, self.xG_factor)

    def __repr__(self):
        return f"Match({self.home_team.name} vs {self.away_team.name} on {self.date})"

    def get_match_expectation(self, teams, xG_factor=0.6):
        league_avg_home, league_avg_away = Team.get_league_averages(teams, xG_factor)
        home_expected_goals = league_avg_home * self.home_team.home_attack_strength * self.away_team.away_defence_strength
        away_expected_goals = league_avg_away * self.away_team.away_attack_strength * self.home_team.home_defence_strength
        
        return home_expected_goals, away_expected_goals

    @classmethod
    def from_fixtures(cls, teams, fixtures, xG_factor, max_goals):
        matches = []
        for _, fixture in fixtures.iterrows():
            match = cls(teams, fixture, xG_factor, max_goals) if 'max_goals' in cls.__init__.__code__.co_varnames else cls(teams, fixture, xG_factor)
            matches.append(match)
        return matches

class MarketsMatch(Match):
    def __init__(self, teams, fixture, xG_factor=0.6, max_goals=6):
        super().__init__(teams, fixture, xG_factor)
        self.max_goals = max_goals
        self.score_matrix = self.get_score_matrix()
        self.markets = self.get_match_markets()

    def get_score_matrix(self):
        home_distribution = poisson.pmf(np.arange(self.max_goals + 1), self.match_expectation[0])
        away_distribution = poisson.pmf(np.arange(self.max_goals + 1), self.match_expectation[1])
        score_matrix = np.outer(home_distribution, away_distribution)

        score_matrix[-1,:-1] += (1-home_distribution.sum()) * away_distribution[:-1]
        score_matrix[:-1,-1] += home_distribution[:-1] * (1-away_distribution.sum())
        score_matrix[-1,-1] += (1-home_distribution.sum()) * (1-away_distribution.sum())

        return score_matrix
    
    def get_match_markets(self):
        # Calculate probabilities for each outcome
        home_win = np.tril(self.score_matrix,-1).sum()
        draw = np.trace(self.score_matrix)
        away_win = np.triu(self.score_matrix, 1).sum()

        # Both Teams to Score
        btts = 1- (self.score_matrix[0,:].sum() + self.score_matrix[:,0].sum() - self.score_matrix[0,0])

        #Over/under 2.5 Goals
        total_goals = np.add.outer(np.arange(self.score_matrix.shape[0]), np.arange(self.score_matrix.shape[1]))
        over_2_5  = self.score_matrix[total_goals > 2.5].sum()
        under_2_5 = 1 - over_2_5

        return {
        "P(Home Win)": home_win,
        "P(Draw)"   : draw,
        "P(Away Win)": away_win,
        "P(BTTS)"   : btts,
        "P(Over 2.5 Goals)": over_2_5,
        "P(Under 2.5 Goals)": under_2_5
    }

class SimmedMatch(Match):
    def __init__(self, teams, fixture, xg_factor=0.6):
        super().__init__(teams, fixture, xg_factor)
        self.sim_result = self.get_sim_result()

    def get_sim_result(self):
        home_exp, away_exp = self.match_expectation
        
        rng = np.random.default_rng()
        home_goals = rng.poisson(home_exp)
        away_goals = rng.poisson(away_exp)
        home_points, away_points = get_points(home_goals, away_goals)
        
        trial_result = ({
                "Date": self.date,
                "Home": self.home_team.name,
                "Away": self.away_team.name,
                "HomeGoals": home_goals,
                "AwayGoals": away_goals,
                "Home_xG": home_exp,
                "Away_xG": away_exp,
                "Home_pts": home_points,
                "Away_pts": away_points
            })
        return trial_result