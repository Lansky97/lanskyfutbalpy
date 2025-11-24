import unittest
import numpy as np
import pandas as pd
from team import Team
from league import League
from match import Match, SimmedMatch

class TestTeamStrength(unittest.TestCase):
    def setUp(self):
        # Minimal valid results
        self.results = [
            {'Date': '2024-08-01', 'Home': 'A', 'Away': 'B', 'HomeGoals': 2, 'AwayGoals': 1, 'Home_xG': 1.5, 'Away_xG': 1.2, 'Home_pts': 3, 'Away_pts': 0},
            {'Date': '2024-08-02', 'Home': 'B', 'Away': 'A', 'HomeGoals': 0, 'AwayGoals': 2, 'Home_xG': 0.8, 'Away_xG': 1.7, 'Home_pts': 0, 'Away_pts': 3}
        ]
        self.teams = Team.teams_from_results(pd.DataFrame(self.results), xG_factor=0.6)

    def test_strengths_calculated(self):
        for team in self.teams.values():
            self.assertIsInstance(team.home_attack_strength, float)
            self.assertIsInstance(team.away_attack_strength, float)
            self.assertGreaterEqual(team.home_attack_strength, 0)
            self.assertGreaterEqual(team.away_attack_strength, 0)

class TestFixtureSimulation(unittest.TestCase):
    def setUp(self):
        results = [
            {'Date': '2024-08-01', 'Home': 'A', 'Away': 'B', 'HomeGoals': 2, 'AwayGoals': 1, 'Home_xG': 1.5, 'Away_xG': 1.2, 'Home_pts': 3, 'Away_pts': 0},
            {'Date': '2024-08-02', 'Home': 'B', 'Away': 'A', 'HomeGoals': 0, 'AwayGoals': 2, 'Home_xG': 0.8, 'Away_xG': 1.7, 'Home_pts': 0, 'Away_pts': 3}
        ]
        self.teams = Team.teams_from_results(pd.DataFrame(results), xG_factor=0.6)
        self.fixture = {'Date': '2024-08-03', 'Home': 'A', 'Away': 'B'}
        self.xG_factor = 0.6
        self.rng = np.random.default_rng(42)

    def test_simmed_match(self):
        match = SimmedMatch(self.teams, self.fixture, self.xG_factor, self.rng)
        result = match.sim_result
        self.assertIn('HomeGoals', result)
        self.assertIn('AwayGoals', result)
        self.assertIsInstance(result['HomeGoals'], int)
        self.assertIsInstance(result['AwayGoals'], int)

class TestTableAggregation(unittest.TestCase):
    def setUp(self):
        self.matches = [
            {'Date': '2024-08-01', 'Home': 'A', 'Away': 'B', 'HomeGoals': 2, 'AwayGoals': 1, 'Home_xG': 1.5, 'Away_xG': 1.2, 'Home_pts': 3, 'Away_pts': 0},
            {'Date': '2024-08-02', 'Home': 'B', 'Away': 'A', 'HomeGoals': 0, 'AwayGoals': 2, 'Home_xG': 0.8, 'Away_xG': 1.7, 'Home_pts': 0, 'Away_pts': 3}
        ]
        self.league = League('TestLeague', self.matches, '2024-12-01', xG_factor=0.6)
        self.league.teams = Team.teams_from_results(self.league.results, xG_factor=0.6)

    def test_league_table(self):
        table = self.league.generate_league_table()
        self.assertIn('Team', table.columns)
        self.assertIn('Points', table.columns)
        self.assertEqual(len(table), 2)
        # Edge case: all teams have same points
        table_sorted = table.sort_values(by=['Points', 'Goal Difference', 'Goals'], ascending=False)
        self.assertEqual(list(table_sorted['Pos']), [1,2])

if __name__ == '__main__':
    unittest.main()
