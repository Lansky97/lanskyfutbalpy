import numpy as np
from scipy.stats import poisson
class Match:
    def __init__(self, date, home_team, away_team, league_averages, max_goals=5):
        self.date = date
        self.home_team = home_team
        self.away_team = away_team
        self.max_goals = max_goals
        self.league_avg_home = league_averages[0]
        self.league_avg_away = league_averages[1]
        self.match_expectation = self.get_match_expectation() 
        self.score_matrix = self.get_score_matrix()
        self.markets = self.get_match_probabilities()

    def __repr__(self):
        return f"Match({self.home_team.name} vs {self.away_team.name})"

    def get_match_expectation(self):
        home_expected_goals = self.league_avg_home * self.home_team.home_attack_strength * self.away_team.away_defence_strength
        away_expected_goals = self.league_avg_away * self.away_team.away_attack_strength * self.home_team.home_defence_strength
        
        return home_expected_goals, away_expected_goals
    
    def sample_result(self, n_trials=1):
        home_exp, away_exp = self.match_expectation
        
        rng = np.random.default_rng()
        home_goals = rng.poisson(home_exp, n_trials)
        away_goals = rng.poisson(away_exp, n_trials)

        trial_results = []
        for i in range(n_trials):
            hg = home_goals[i]
            ag = away_goals[i]
            if hg > ag:
                home_points, away_points = 3, 0
            elif hg < ag:
                home_points, away_points = 0, 3
            else:
                home_points, away_points = 1, 1

            trial_results.append({
                "Date": self.date,
                "Home": self.home_team.name,
                "Away": self.away_team.name,
                "HomeGoals": hg,
                "AwayGoals": ag,
                "Home_xG": home_exp,
                "Away_xG": away_exp,
                "Home_pts": home_points,
                "Away_pts": away_points
            })
        return trial_results

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