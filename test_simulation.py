import unittest
import pandas as pd
from team import Team
from league import League
from simulation import Config, Simulation


def _make_league():
    """Build a minimal 4-team League with past results and future fixtures."""
    matches = [
        {'Date': '2024-08-01', 'Home': 'A', 'Away': 'B', 'HomeGoals': 2, 'AwayGoals': 1, 'Home_xG': 1.5, 'Away_xG': 1.2},
        {'Date': '2024-08-01', 'Home': 'C', 'Away': 'D', 'HomeGoals': 1, 'AwayGoals': 1, 'Home_xG': 1.1, 'Away_xG': 1.1},
        {'Date': '2024-08-08', 'Home': 'B', 'Away': 'C', 'HomeGoals': 0, 'AwayGoals': 1, 'Home_xG': 0.9, 'Away_xG': 1.3},
        {'Date': '2024-08-08', 'Home': 'D', 'Away': 'A', 'HomeGoals': 2, 'AwayGoals': 2, 'Home_xG': 1.4, 'Away_xG': 1.6},
        {'Date': '2024-10-01', 'Home': 'A', 'Away': 'C', 'HomeGoals': None, 'AwayGoals': None, 'Home_xG': None, 'Away_xG': None},
        {'Date': '2024-10-01', 'Home': 'B', 'Away': 'D', 'HomeGoals': None, 'AwayGoals': None, 'Home_xG': None, 'Away_xG': None},
        {'Date': '2024-10-08', 'Home': 'C', 'Away': 'A', 'HomeGoals': None, 'AwayGoals': None, 'Home_xG': None, 'Away_xG': None},
        {'Date': '2024-10-08', 'Home': 'D', 'Away': 'B', 'HomeGoals': None, 'AwayGoals': None, 'Home_xG': None, 'Away_xG': None},
        {'Date': '2024-10-15', 'Home': 'A', 'Away': 'D', 'HomeGoals': None, 'AwayGoals': None, 'Home_xG': None, 'Away_xG': None},
        {'Date': '2024-10-15', 'Home': 'B', 'Away': 'C', 'HomeGoals': None, 'AwayGoals': None, 'Home_xG': None, 'Away_xG': None},
    ]
    lge = League('TestLeague', matches, '2024-09-01', xG_factor=0.6)
    lge.teams = Team.teams_from_results(
        lge.results, lge.league_avg_home, lge.league_avg_away, xG_factor=0.6
    )
    lge.league_table = lge.generate_league_table()
    return lge


class TestConfigInit(unittest.TestCase):
    def test_seed_stored(self):
        """Config stores the seed passed to the constructor."""
        self.assertEqual(Config(seed=42).seed, 42)

    def test_seed_defaults_to_none(self):
        """Config seed defaults to None when not provided."""
        self.assertIsNone(Config().seed)


class TestSimulationInit(unittest.TestCase):
    def setUp(self):
        self.league = _make_league()

    def test_non_league_raises_type_error(self):
        """Passing a non-League object as league raises TypeError."""
        with self.assertRaises(TypeError):
            Simulation("not a league", 10, Config(seed=42))

    def test_zero_n_trials_raises_value_error(self):
        """n_trials=0 raises ValueError."""
        with self.assertRaises(ValueError):
            Simulation(self.league, 0, Config(seed=42))

    def test_negative_n_trials_raises_value_error(self):
        """Negative n_trials raises ValueError."""
        with self.assertRaises(ValueError):
            Simulation(self.league, -1, Config(seed=42))

    def test_float_n_trials_raises_value_error(self):
        """Float n_trials raises ValueError because the int isinstance check fails."""
        with self.assertRaises(ValueError):
            Simulation(self.league, 1.5, Config(seed=42))


class TestSimulateSeason(unittest.TestCase):
    def setUp(self):
        self.league = _make_league()
        self.sim = Simulation(self.league, n_trials=20, config=Config(seed=42))

    def test_returns_n_trials_leagues(self):
        """simulate_season produces exactly n_trials trial League objects."""
        self.assertEqual(len(self.sim.simmed_leagues), 20)

    def test_reproducible_with_same_seed(self):
        """Two Simulations built from identical leagues with the same seed produce identical mean points per team."""
        sim2 = Simulation(_make_league(), n_trials=20, config=Config(seed=42))
        pts1 = self.sim.mean_final_table.set_index('Team')['Points'].to_dict()
        pts2 = sim2.mean_final_table.set_index('Team')['Points'].to_dict()
        self.assertEqual(pts1, pts2)


class TestLeagueSnapshotRestore(unittest.TestCase):
    def setUp(self):
        self.league = _make_league()
        self.sim = Simulation(self.league, n_trials=5, config=Config(seed=42))

    def test_snapshot_captures_all_expected_fields(self):
        """_league_snapshot dict contains every expected league-level key."""
        snap = self.sim._league_snapshot()
        for key in ('teams', 'results_len', 'total_home_goals', 'total_away_goals',
                    'total_home_xg', 'total_away_xg', 'league_avg_home',
                    'league_avg_away', 'games_played'):
            with self.subTest(key=key):
                self.assertIn(key, snap)

    def test_restore_reverts_modified_league_fields(self):
        """_league_restore correctly reverts all league scalar fields to their snapshot values."""
        snap = self.sim._league_snapshot()
        pre_home_goals = self.league.total_home_goals
        pre_avg_home = self.league.league_avg_home
        pre_results_len = len(self.league.results)

        self.league.total_home_goals += 999
        self.league.league_avg_home += 99.0
        self.league.results.extend([{'dummy': True}])

        self.sim._league_restore(snap)

        self.assertEqual(self.league.total_home_goals, pre_home_goals)
        self.assertAlmostEqual(self.league.league_avg_home, pre_avg_home)
        self.assertEqual(len(self.league.results), pre_results_len)


class TestTeamsSnapshot(unittest.TestCase):
    FIELDS = (
        'home_games_played', 'away_games_played',
        'home_points', 'away_points',
        'home_goals', 'away_goals',
        'home_goals_against', 'away_goals_against',
        'home_xg', 'away_xg', 'home_xga', 'away_xga',
        'home_attack_strength_cs', 'away_attack_strength_cs',
        'home_defence_strength_cs', 'away_defence_strength_cs',
        'home_attack_strength', 'away_attack_strength',
        'home_defence_strength', 'away_defence_strength',
    )

    def setUp(self):
        self.league = _make_league()
        self.sim = Simulation(self.league, n_trials=5, config=Config(seed=42))

    def test_snapshot_contains_all_fields_for_every_team(self):
        """_teams_snapshot includes every expected Team stat field for each team."""
        snap = self.sim._teams_snapshot(self.league.teams)
        for name in self.league.teams:
            for field in self.FIELDS:
                with self.subTest(team=name, field=field):
                    self.assertIn(field, snap[name])

    def test_restore_teams_reverts_all_fields(self):
        """_restore_teams puts every captured stat field back to its pre-mutation value."""
        snap = self.sim._teams_snapshot(self.league.teams)
        team_a = self.league.teams['A']
        orig_goals = team_a.home_goals
        orig_strength = team_a.home_attack_strength

        team_a.home_goals += 999
        team_a.home_attack_strength += 99.0

        self.sim._restore_teams(self.league.teams, snap)

        self.assertEqual(team_a.home_goals, orig_goals)
        self.assertAlmostEqual(team_a.home_attack_strength, orig_strength)


class TestMeanFinalTable(unittest.TestCase):
    def setUp(self):
        self.league = _make_league()
        self.sim = Simulation(self.league, n_trials=30, config=Config(seed=42))

    def test_returns_dataframe(self):
        """mean_final_table is a pandas DataFrame."""
        self.assertIsInstance(self.sim.mean_final_table, pd.DataFrame)

    def test_contains_all_teams(self):
        """mean_final_table has exactly one row per team."""
        self.assertEqual(set(self.sim.mean_final_table['Team']), set(self.league.teams))

    def test_pos_column_runs_1_to_n(self):
        """Pos column is a sequential 1..n with no gaps or duplicates."""
        pos = sorted(self.sim.mean_final_table['Pos'].tolist())
        self.assertEqual(pos, list(range(1, len(self.league.teams) + 1)))

    def test_sorted_by_points_descending(self):
        """Rows are ordered by mean Points descending so Pos 1 has the highest points."""
        pts = self.sim.mean_final_table['Points'].tolist()
        self.assertEqual(pts, sorted(pts, reverse=True))


class TestPositionOdds(unittest.TestCase):
    def setUp(self):
        self.league = _make_league()
        self.sim = Simulation(self.league, n_trials=50, config=Config(seed=42))

    def test_each_team_probs_sum_to_approx_1(self):
        """For every team, probabilities across all positions sum to approximately 1.0."""
        for team in self.sim.position_odds.index:
            with self.subTest(team=team):
                self.assertAlmostEqual(self.sim.position_odds.loc[team].sum(), 1.0, places=5)


class TestCompetitionMarkets(unittest.TestCase):
    def setUp(self):
        self.league = _make_league()
        self.sim = Simulation(self.league, n_trials=50, config=Config(seed=42))

    def test_champion_matches_position_1_odds(self):
        """Champion probability for each team equals the position-1 column in position_odds."""
        for team in self.sim.competition_markets.index:
            with self.subTest(team=team):
                self.assertAlmostEqual(
                    self.sim.competition_markets.loc[team, 'Champion'],
                    self.sim.position_odds.loc[team, 1],
                    places=10
                )

    def test_relegation_sums_bottom_3_positions(self):
        """Relegation probability equals the sum of the bottom 3 position columns in position_odds."""
        num_teams = len(self.league.teams)
        for team in self.sim.competition_markets.index:
            with self.subTest(team=team):
                expected = sum(
                    self.sim.position_odds.loc[team, pos]
                    for pos in [num_teams - 2, num_teams - 1, num_teams]
                )
                self.assertAlmostEqual(
                    self.sim.competition_markets.loc[team, 'Relegation'],
                    expected,
                    places=10
                )


class TestNonDestructiveSimulation(unittest.TestCase):
    def test_league_state_identical_before_and_after_simulation(self):
        """Running a Simulation leaves the input League state completely unchanged."""
        league = _make_league()

        pre_results_len = len(league.results)
        pre_games_played = league.games_played
        pre_home_goals = league.total_home_goals
        pre_avg_home = league.league_avg_home
        pre_a_goals = league.teams['A'].home_goals
        pre_a_strength = league.teams['A'].home_attack_strength

        Simulation(league, n_trials=10, config=Config(seed=42))

        self.assertEqual(len(league.results), pre_results_len)
        self.assertEqual(league.games_played, pre_games_played)
        self.assertEqual(league.total_home_goals, pre_home_goals)
        self.assertAlmostEqual(league.league_avg_home, pre_avg_home)
        self.assertEqual(league.teams['A'].home_goals, pre_a_goals)
        self.assertAlmostEqual(league.teams['A'].home_attack_strength, pre_a_strength)


if __name__ == '__main__':
    unittest.main()
