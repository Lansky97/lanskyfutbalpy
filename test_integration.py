import math
import unittest
import pandas as pd
import numpy as np
from team import Team
from league import League
from simulation import Config, Simulation
from evaluate import Evaluate


# ---------------------------------------------------------------------------
# Shared fixtures — minimal 4-team dataset, no file/DB access
# ---------------------------------------------------------------------------

_MATCHES = [
    {"Date": "2024-08-01", "Home": "A", "Away": "B", "HomeGoals": 2, "AwayGoals": 1, "Home_xG": 1.5, "Away_xG": 1.2},
    {"Date": "2024-08-01", "Home": "C", "Away": "D", "HomeGoals": 1, "AwayGoals": 1, "Home_xG": 1.1, "Away_xG": 1.1},
    {"Date": "2024-08-08", "Home": "B", "Away": "C", "HomeGoals": 0, "AwayGoals": 1, "Home_xG": 0.9, "Away_xG": 1.3},
    {"Date": "2024-08-08", "Home": "D", "Away": "A", "HomeGoals": 2, "AwayGoals": 2, "Home_xG": 1.4, "Away_xG": 1.6},
    {"Date": "2024-08-15", "Home": "A", "Away": "D", "HomeGoals": 3, "AwayGoals": 0, "Home_xG": 2.1, "Away_xG": 0.8},
    {"Date": "2024-08-15", "Home": "C", "Away": "B", "HomeGoals": 2, "AwayGoals": 2, "Home_xG": 1.3, "Away_xG": 1.4},
    {"Date": "2024-10-01", "Home": "A", "Away": "C", "HomeGoals": None, "AwayGoals": None, "Home_xG": None, "Away_xG": None},
    {"Date": "2024-10-01", "Home": "B", "Away": "D", "HomeGoals": None, "AwayGoals": None, "Home_xG": None, "Away_xG": None},
    {"Date": "2024-10-08", "Home": "D", "Away": "C", "HomeGoals": None, "AwayGoals": None, "Home_xG": None, "Away_xG": None},
    {"Date": "2024-10-08", "Home": "B", "Away": "A", "HomeGoals": None, "AwayGoals": None, "Home_xG": None, "Away_xG": None},
]

_ACTUAL_FINAL_TABLE = pd.DataFrame({
    "Team":   ["A",  "C",  "D", "B"],
    "Pos":    [1,     2,     3,    4],
    "Points": [10.0,  8.0,   5.0,  4.0],
})


def _make_league():
    lge = League("TestLeague", _MATCHES, "2024-09-01", xG_factor=0.6)
    lge.teams = Team.teams_from_results(
        lge.results, lge.league_avg_home, lge.league_avg_away, xG_factor=0.6
    )
    lge.league_table = lge.generate_league_table()
    return lge


# ---------------------------------------------------------------------------
# League → Simulation
# ---------------------------------------------------------------------------

class TestLeagueToSimulationIntegration(unittest.TestCase):
    def test_league_state_unchanged_after_full_simulation(self):
        """Running a full Simulation leaves every observable League field intact."""
        league = _make_league()
        pre_results_len = len(league.results)
        pre_games_played = league.games_played
        pre_home_goals = league.total_home_goals
        pre_avg_home = league.league_avg_home
        pre_a_strength = league.teams["A"].home_attack_strength

        Simulation(league, n_trials=50, config=Config(seed=42))

        self.assertEqual(len(league.results), pre_results_len)
        self.assertEqual(league.games_played, pre_games_played)
        self.assertEqual(league.total_home_goals, pre_home_goals)
        self.assertAlmostEqual(league.league_avg_home, pre_avg_home)
        self.assertAlmostEqual(league.teams["A"].home_attack_strength, pre_a_strength)

    def test_simmed_leagues_count_matches_n_trials(self):
        """Simulation produces exactly n_trials completed League objects."""
        league = _make_league()
        sim = Simulation(league, n_trials=50, config=Config(seed=42))
        self.assertEqual(len(sim.simmed_leagues), 50)

    def test_every_simmed_league_table_has_all_teams(self):
        """Each trial League table contains exactly the same set of teams as the source League."""
        league = _make_league()
        sim = Simulation(league, n_trials=20, config=Config(seed=42))
        expected_teams = set(league.teams)
        for i, trial in enumerate(sim.simmed_leagues):
            with self.subTest(trial=i):
                trial_teams = {row["Team"] for row in trial.league_table}
                self.assertEqual(trial_teams, expected_teams)


# ---------------------------------------------------------------------------
# Simulation → Evaluate
# ---------------------------------------------------------------------------

class TestSimulationToEvaluateIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        league = _make_league()
        cls.sim = Simulation(league, n_trials=100, config=Config(seed=42))
        cls.ev = Evaluate(cls.sim, _ACTUAL_FINAL_TABLE)

    def test_evaluate_constructs_without_error(self):
        """Evaluate accepts a real Simulation object and constructs without raising."""
        self.assertIsNotNone(self.ev)

    def test_probs_df_shape_matches_team_count(self):
        """probs_df produced from a real Simulation has shape (n_teams, n_teams)."""
        n = len(_ACTUAL_FINAL_TABLE)
        self.assertEqual(self.ev.probs["probs_df"].shape, (n, n))

    def test_metrics_report_columns_correct(self):
        """metrics_report from a real pipeline has exactly the three expected columns."""
        self.assertEqual(
            list(self.ev.metrics_report.columns),
            ["Metric Group", "Metric", "Value"]
        )

    def test_metrics_report_row_count(self):
        """metrics_report contains exactly 9 rows (3 proper + 2 ranking + 4 points)."""
        self.assertEqual(len(self.ev.metrics_report), 9)

    def test_all_metric_values_are_finite(self):
        """Every metric value produced from a real Simulation→Evaluate pipeline is finite."""
        for _, row in self.ev.metrics_report.iterrows():
            with self.subTest(metric=row["Metric"]):
                self.assertTrue(math.isfinite(float(row["Value"])))

    def test_position_odds_index_aligns_with_actual_teams(self):
        """Real Simulation.position_odds index contains all teams from the actual final table."""
        actual_teams = set(_ACTUAL_FINAL_TABLE["Team"])
        odds_teams = set(self.sim.position_odds.index)
        self.assertTrue(actual_teams.issubset(odds_teams))

    def test_mean_final_table_has_required_columns(self):
        """Real Simulation.mean_final_table has Team, Pos, and Points columns for Evaluate."""
        for col in ("Team", "Pos", "Points"):
            with self.subTest(col=col):
                self.assertIn(col, self.sim.mean_final_table.columns)


# ---------------------------------------------------------------------------
# Full pipeline: League → Simulation → Evaluate
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        league = _make_league()
        cls.sim = Simulation(league, n_trials=100, config=Config(seed=42))
        cls.ev = Evaluate(cls.sim, _ACTUAL_FINAL_TABLE)

    def test_rps_in_valid_range(self):
        """RPS from the full pipeline is in the valid range [0.0, 1.0]."""
        rps = self.ev.get_ranked_probability_score()
        self.assertGreaterEqual(rps, 0.0)
        self.assertLessEqual(rps, 1.0)

    def test_brier_score_in_valid_range(self):
        """Brier score from the full pipeline is in the valid range [0.0, 1.0]."""
        brier = self.ev.get_brier_score()
        self.assertGreaterEqual(brier, 0.0)
        self.assertLessEqual(brier, 1.0)

    def test_position_odds_sum_to_one_per_team(self):
        """Real Simulation.position_odds rows each sum to 1.0 (valid probability distribution)."""
        for team in self.sim.position_odds.index:
            with self.subTest(team=team):
                self.assertAlmostEqual(
                    self.sim.position_odds.loc[team].sum(), 1.0, places=5
                )

    def test_pipeline_reproducibility_with_same_seed(self):
        """Two identical pipelines with the same seed produce identical metrics_report values."""
        def run():
            league = _make_league()
            sim = Simulation(league, n_trials=50, config=Config(seed=42))
            return Evaluate(sim, _ACTUAL_FINAL_TABLE).metrics_report["Value"].tolist()

        self.assertEqual(run(), run())

    def test_metrics_report_metric_order(self):
        """metrics_report rows appear in the custom order defined by the order dict (not alphabetical)."""
        report = self.ev.metrics_report
        proper_rows = report[report["Metric Group"] == "PROPER"]["Metric"].tolist()
        self.assertEqual(proper_rows, ["rps", "brier_score", "log_loss"])


if __name__ == "__main__":
    unittest.main()
