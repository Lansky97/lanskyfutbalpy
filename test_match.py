import unittest
import numpy as np
from team import Team
from match import Match, MarketsMatch, SimmedMatch


def _make_team(name, has=1.0, hds=1.0, aas=1.0, ads=1.0):
    t = Team(name)
    t.home_attack_strength = has
    t.home_defence_strength = hds
    t.away_attack_strength = aas
    t.away_defence_strength = ads
    return t


LEAGUE_AVG_HOME = 1.5
LEAGUE_AVG_AWAY = 1.2


class TestMatchInit(unittest.TestCase):
    def setUp(self):
        self.teams = {'A': _make_team('A'), 'B': _make_team('B')}
        self.fixture = {'Date': '2024-08-01', 'Home': 'A', 'Away': 'B'}

    def test_missing_date_raises(self):
        """Missing 'Date' key in fixture raises ValueError."""
        with self.assertRaises(ValueError):
            Match(self.teams, {'Home': 'A', 'Away': 'B'}, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)

    def test_missing_home_raises(self):
        """Missing 'Home' key in fixture raises ValueError."""
        with self.assertRaises(ValueError):
            Match(self.teams, {'Date': '2024-08-01', 'Away': 'B'}, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)

    def test_missing_away_raises(self):
        """Missing 'Away' key in fixture raises ValueError."""
        with self.assertRaises(ValueError):
            Match(self.teams, {'Date': '2024-08-01', 'Home': 'A'}, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)

    def test_unknown_home_team_raises(self):
        """Home team not in teams dict raises ValueError."""
        with self.assertRaises(ValueError):
            Match(self.teams, {'Date': '2024-08-01', 'Home': 'X', 'Away': 'B'}, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)

    def test_unknown_away_team_raises(self):
        """Away team not in teams dict raises ValueError."""
        with self.assertRaises(ValueError):
            Match(self.teams, {'Date': '2024-08-01', 'Home': 'A', 'Away': 'X'}, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)

    def test_valid_fixture_stores_date_and_teams(self):
        """Valid fixture creates Match with correct date and team name references."""
        m = Match(self.teams, self.fixture, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)
        self.assertEqual(m.date, '2024-08-01')
        self.assertEqual(m.home_team.name, 'A')
        self.assertEqual(m.away_team.name, 'B')


class TestGetMatchExpectation(unittest.TestCase):
    def test_exact_expected_goals(self):
        """Expected goals = avg_home * home_atk * away_def (home) and avg_away * away_atk * home_def (away)."""
        # home_exp = 1.5 * 1.3 * 0.9 = 1.755
        # away_exp = 1.2 * 0.8 * 1.1 = 1.056
        home = _make_team('H', has=1.3, hds=1.1)
        away = _make_team('A', aas=0.8, ads=0.9)
        m = Match({'H': home, 'A': away}, {'Date': '2024-08-01', 'Home': 'H', 'Away': 'A'}, 1.5, 1.2, 0.6)
        home_exp, away_exp = m.match_expectation
        self.assertAlmostEqual(home_exp, 1.5 * 1.3 * 0.9, places=10)
        self.assertAlmostEqual(away_exp, 1.2 * 0.8 * 1.1, places=10)


class TestMarketsMatchScoreMatrix(unittest.TestCase):
    def setUp(self):
        # home_exp = 1.5 * 1.0 * 0.9 = 1.35; away_exp = 1.2 * 0.8 * 1.0 = 0.96
        home = _make_team('H', has=1.0, hds=1.0)
        away = _make_team('A', aas=0.8, ads=0.9)
        self.match = MarketsMatch(
            {'H': home, 'A': away},
            {'Date': '2024-08-01', 'Home': 'H', 'Away': 'A'},
            LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6
        )

    def test_score_matrix_sums_to_approx_1(self):
        """Score matrix total probability sums to approximately 1.0."""
        self.assertAlmostEqual(self.match.score_matrix.sum(), 1.0, places=5)

    def test_score_matrix_shape(self):
        """Default max_goals=6 produces a (7, 7) score matrix."""
        self.assertEqual(self.match.score_matrix.shape, (7, 7))

    def test_score_matrix_all_non_negative(self):
        """Every entry in the score matrix is non-negative."""
        self.assertTrue((self.match.score_matrix >= 0.0).all())


class TestMarketsMatchGetMatchMarkets(unittest.TestCase):
    def setUp(self):
        home = _make_team('H', has=1.0, hds=1.0)
        away = _make_team('A', aas=0.8, ads=0.9)
        self.match = MarketsMatch(
            {'H': home, 'A': away},
            {'Date': '2024-08-01', 'Home': 'H', 'Away': 'A'},
            LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6
        )

    def test_all_probabilities_between_0_and_1(self):
        """Every market probability lies in [0.0, 1.0]."""
        for key, val in self.match.markets.items():
            with self.subTest(key=key):
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_home_draw_away_sums_to_1(self):
        """P(Home Win) + P(Draw) + P(Away Win) sums to approximately 1.0."""
        m = self.match.markets
        total = m['P(Home Win)'] + m['P(Draw)'] + m['P(Away Win)']
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_over_under_sums_to_1(self):
        """P(Over 2.5 Goals) + P(Under 2.5 Goals) = 1.0 (Under is defined as 1 - Over)."""
        m = self.match.markets
        total = m['P(Over 2.5 Goals)'] + m['P(Under 2.5 Goals)']
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_required_keys_present(self):
        """markets dict contains exactly the 6 expected probability keys."""
        expected = {'P(Home Win)', 'P(Draw)', 'P(Away Win)', 'P(BTTS)', 'P(Over 2.5 Goals)', 'P(Under 2.5 Goals)'}
        self.assertEqual(set(self.match.markets.keys()), expected)


class TestSimmedMatchGetSimResult(unittest.TestCase):
    def setUp(self):
        home = _make_team('H', has=1.2, hds=1.0)
        away = _make_team('A', aas=0.8, ads=1.1)
        self.teams = {'H': home, 'A': away}
        self.fixture = {'Date': '2024-08-01', 'Home': 'H', 'Away': 'A'}
        self.rng = np.random.default_rng(42)
        self.match = SimmedMatch(self.teams, self.fixture, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6, self.rng)

    def test_required_keys_present(self):
        """sim_result contains all 9 required keys."""
        required = {'Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Home_xG', 'Away_xG', 'Home_pts', 'Away_pts'}
        self.assertEqual(set(self.match.sim_result.keys()), required)

    def test_goal_values_are_integers(self):
        """HomeGoals and AwayGoals are integer type (Python int or numpy integer)."""
        result = self.match.sim_result
        self.assertIsInstance(result['HomeGoals'], (int, np.integer))
        self.assertIsInstance(result['AwayGoals'], (int, np.integer))

    def test_points_consistent_with_goals(self):
        """Points match the goal outcome: home win → 3/0, draw → 1/1, away win → 0/3."""
        result = self.match.sim_result
        hg, ag = result['HomeGoals'], result['AwayGoals']
        hp, ap = result['Home_pts'], result['Away_pts']
        if hg > ag:
            self.assertEqual(hp, 3)
            self.assertEqual(ap, 0)
        elif hg < ag:
            self.assertEqual(hp, 0)
            self.assertEqual(ap, 3)
        else:
            self.assertEqual(hp, 1)
            self.assertEqual(ap, 1)

    def test_reproducible_with_fixed_seed(self):
        """Identical RNG seed always produces the same goal counts."""
        rng2 = np.random.default_rng(42)
        match2 = SimmedMatch(self.teams, self.fixture, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6, rng2)
        self.assertEqual(self.match.sim_result['HomeGoals'], match2.sim_result['HomeGoals'])
        self.assertEqual(self.match.sim_result['AwayGoals'], match2.sim_result['AwayGoals'])

    def test_xg_equals_match_expectation(self):
        """Home_xG and Away_xG store the Poisson lambdas, not randomly sampled values."""
        home_exp, away_exp = self.match.match_expectation
        self.assertAlmostEqual(self.match.sim_result['Home_xG'], home_exp)
        self.assertAlmostEqual(self.match.sim_result['Away_xG'], away_exp)

    def test_team_names_in_result(self):
        """Home and Away names in sim_result match the fixture team names."""
        self.assertEqual(self.match.sim_result['Home'], 'H')
        self.assertEqual(self.match.sim_result['Away'], 'A')


class TestMatchFromFixtures(unittest.TestCase):
    def setUp(self):
        home = _make_team('H', has=1.2, hds=1.0)
        away = _make_team('A', aas=0.8, ads=1.1)
        self.teams = {'H': home, 'A': away}
        self.fixtures = [
            {'Date': '2024-08-01', 'Home': 'H', 'Away': 'A'},
            {'Date': '2024-08-08', 'Home': 'A', 'Away': 'H'},
        ]

    def test_raises_on_non_list(self):
        """from_fixtures raises TypeError when fixtures is not a list."""
        with self.assertRaises(TypeError):
            Match.from_fixtures(self.teams, "not a list", LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)

    def test_raises_on_non_dict_elements(self):
        """from_fixtures raises TypeError when fixture elements are not dicts."""
        with self.assertRaises(TypeError):
            Match.from_fixtures(self.teams, ["not a dict"], LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)

    def test_dispatches_to_simmed_match(self):
        """SimmedMatch.from_fixtures returns a list of SimmedMatch instances."""
        rng = np.random.default_rng(42)
        matches = SimmedMatch.from_fixtures(self.teams, self.fixtures, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6, rng=rng)
        self.assertEqual(len(matches), 2)
        for m in matches:
            self.assertIsInstance(m, SimmedMatch)

    def test_dispatches_to_markets_match(self):
        """MarketsMatch.from_fixtures returns a list of MarketsMatch instances."""
        matches = MarketsMatch.from_fixtures(self.teams, self.fixtures, LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)
        self.assertEqual(len(matches), 2)
        for m in matches:
            self.assertIsInstance(m, MarketsMatch)

    def test_empty_fixtures_returns_empty_list(self):
        """from_fixtures with an empty list returns an empty list."""
        matches = Match.from_fixtures(self.teams, [], LEAGUE_AVG_HOME, LEAGUE_AVG_AWAY, 0.6)
        self.assertEqual(matches, [])


if __name__ == '__main__':
    unittest.main()
