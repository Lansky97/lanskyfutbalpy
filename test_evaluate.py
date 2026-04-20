import math
import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from evaluate import Evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(position_odds: pd.DataFrame, mean_final_table: pd.DataFrame):
    sim = MagicMock()
    sim.position_odds = position_odds
    sim.mean_final_table = mean_final_table
    return sim


def _perfect_sim_and_actual():
    """4-team setup where predicted probabilities and points match actuals exactly."""
    teams = ["A", "B", "C", "D"]
    positions = [1, 2, 3, 4]
    points = [80.0, 70.0, 60.0, 50.0]

    position_odds = pd.DataFrame(
        np.eye(4),
        index=teams,
        columns=[1, 2, 3, 4]
    )
    mean_final_table = pd.DataFrame({
        "Team": teams,
        "Pos": positions,
        "Points": points
    })
    actual_table = pd.DataFrame({
        "Team": teams,
        "Pos": positions,
        "Points": points
    })
    return _make_sim(position_odds, mean_final_table), actual_table


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvaluateInit(unittest.TestCase):
    def setUp(self):
        self.sim, self.actual = _perfect_sim_and_actual()
        self.ev = Evaluate(self.sim, self.actual)

    def test_stores_simulation(self):
        """Evaluate stores the passed simulation object as self.simulation."""
        self.assertIs(self.ev.simulation, self.sim)

    def test_stores_actual_final_table(self):
        """Evaluate stores the passed actual_final_table as self.actual_final_table."""
        pd.testing.assert_frame_equal(self.ev.actual_final_table, self.actual)

    def test_probs_populated(self):
        """__init__ populates self.probs dict with probs_df, actual_pos, num_positions."""
        for key in ("probs_df", "actual_pos", "num_positions"):
            with self.subTest(key=key):
                self.assertIn(key, self.ev.probs)

    def test_actuals_populated(self):
        """__init__ populates self.actuals dict with one_hot_actuals and cumulative_actuals."""
        for key in ("one_hot_actuals", "cumulative_actuals"):
            with self.subTest(key=key):
                self.assertIn(key, self.ev.actuals)

    def test_aligned_positions_and_points_populated(self):
        """__init__ populates self.aligned_positions_and_points with four array keys."""
        for key in ("position_actual", "points_actual",
                    "position_prediction", "points_prediction"):
            with self.subTest(key=key):
                self.assertIn(key, self.ev.aligned_positions_and_points)

    def test_metrics_report_is_dataframe(self):
        """__init__ stores self.metrics_report as a pandas DataFrame."""
        self.assertIsInstance(self.ev.metrics_report, pd.DataFrame)


class TestPrepProbs(unittest.TestCase):
    def setUp(self):
        self.sim, self.actual = _perfect_sim_and_actual()
        self.ev = Evaluate(self.sim, self.actual)

    def test_probs_df_shape_is_n_by_n(self):
        """probs_df is a square array with one row and column per team/position."""
        n = len(self.actual)
        self.assertEqual(self.ev.probs["probs_df"].shape, (n, n))

    def test_fills_missing_positions_with_zero(self):
        """Positions absent from simulation.position_odds are filled with 0.0."""
        # position_odds only has columns 1 and 2; actual has 3 teams so column 3 is expected
        partial_odds = pd.DataFrame(
            {1: [0.6, 0.1, 0.3], 2: [0.4, 0.9, 0.7]},
            index=["A", "B", "C"]
        )
        mean_ft = pd.DataFrame({"Team": ["A","B","C"], "Pos": [1,2,3], "Points": [70.,60.,50.]})
        actual = pd.DataFrame({"Team": ["A","B","C"], "Pos": [1,2,3], "Points": [70.,60.,50.]})
        ev = Evaluate(_make_sim(partial_odds, mean_ft), actual)
        np.testing.assert_array_equal(ev.probs["probs_df"][:, 2], [0.0, 0.0, 0.0])

    def test_actual_pos_matches_actual_table(self):
        """actual_pos contains integer positions from actual_final_table in team order."""
        np.testing.assert_array_equal(
            self.ev.probs["actual_pos"], np.array([1, 2, 3, 4])
        )

    def test_teams_aligned_to_actual_table_order(self):
        """probs_df rows follow actual_final_table team order even when position_odds is differently ordered."""
        teams = ["A", "B", "C", "D"]
        # Build position_odds with A last; give A a distinctive value (0.7 at pos 1)
        raw = pd.DataFrame(
            {1: [0.0, 0.0, 0.0, 0.7],
             2: [0.0, 0.0, 0.3, 0.3],
             3: [0.0, 0.4, 0.7, 0.0],
             4: [1.0, 0.6, 0.0, 0.0]},
            index=["D", "C", "B", "A"]
        )
        mean_ft = pd.DataFrame({"Team": teams, "Pos": [1,2,3,4], "Points": [80.,70.,60.,50.]})
        actual = pd.DataFrame({"Team": teams, "Pos": [1,2,3,4], "Points": [80.,70.,60.,50.]})
        ev = Evaluate(_make_sim(raw, mean_ft), actual)
        # Row 0 must be team A (actual order), which has 0.7 at position 1
        self.assertAlmostEqual(ev.probs["probs_df"][0, 0], 0.7)


class TestPrepActualsMatrix(unittest.TestCase):
    def setUp(self):
        self.sim, self.actual = _perfect_sim_and_actual()
        self.ev = Evaluate(self.sim, self.actual)

    def test_one_hot_each_row_sums_to_one(self):
        """Every row of one_hot_actuals sums to 1.0 (exactly one position per team)."""
        row_sums = self.ev.actuals["one_hot_actuals"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4))

    def test_one_hot_entry_at_correct_position(self):
        """one_hot_actuals[i, actual_pos[i]-1] is 1.0 and all other entries in row i are 0."""
        one_hot = self.ev.actuals["one_hot_actuals"]
        actual_pos = self.ev.probs["actual_pos"]
        for i, pos in enumerate(actual_pos):
            with self.subTest(team_index=i):
                self.assertEqual(one_hot[i, pos - 1], 1.0)

    def test_cumulative_actuals_non_decreasing(self):
        """Every row of cumulative_actuals is non-decreasing (a valid step CDF)."""
        cum = self.ev.actuals["cumulative_actuals"]
        for i, row in enumerate(cum):
            with self.subTest(row=i):
                self.assertTrue(np.all(np.diff(row) >= 0))


class TestGetRankedProbabilityScore(unittest.TestCase):
    def test_perfect_prediction_returns_zero(self):
        """RPS is 0.0 when each team is assigned probability 1.0 at their actual position."""
        sim, actual = _perfect_sim_and_actual()
        ev = Evaluate(sim, actual)
        self.assertAlmostEqual(ev.get_ranked_probability_score(), 0.0, places=10)

    def test_imperfect_prediction_is_positive(self):
        """RPS is strictly positive when probabilities are assigned to wrong positions."""
        teams = ["A", "B", "C", "D"]
        # All mass on the wrong position for every team
        position_odds = pd.DataFrame(
            np.eye(4)[::-1],  # A gets all mass on pos 4, D gets all mass on pos 1
            index=teams,
            columns=[1, 2, 3, 4]
        )
        mean_ft = pd.DataFrame({"Team": teams, "Pos": [1,2,3,4], "Points": [80.,70.,60.,50.]})
        actual = pd.DataFrame({"Team": teams, "Pos": [1,2,3,4], "Points": [80.,70.,60.,50.]})
        ev = Evaluate(_make_sim(position_odds, mean_ft), actual)
        self.assertGreater(ev.get_ranked_probability_score(), 0.0)


class TestGetBrierScore(unittest.TestCase):
    def test_perfect_prediction_returns_zero(self):
        """Brier score is 0.0 when predicted probabilities are perfect (1.0 at actual pos)."""
        sim, actual = _perfect_sim_and_actual()
        ev = Evaluate(sim, actual)
        self.assertAlmostEqual(ev.get_brier_score(), 0.0, places=10)

    def test_bounded_between_zero_and_one(self):
        """Brier score lies in [0.0, 1.0] for any valid probability distribution."""
        teams = ["A", "B"]
        position_odds = pd.DataFrame(
            [[0.5, 0.5], [0.5, 0.5]], index=teams, columns=[1, 2]
        )
        mean_ft = pd.DataFrame({"Team": teams, "Pos": [1, 2], "Points": [60., 50.]})
        actual = pd.DataFrame({"Team": teams, "Pos": [1, 2], "Points": [60., 50.]})
        brier = Evaluate(_make_sim(position_odds, mean_ft), actual).get_brier_score()
        self.assertGreaterEqual(brier, 0.0)
        self.assertLessEqual(brier, 1.0)

    def test_imperfect_prediction_is_positive(self):
        """Brier score is positive when predicted probabilities do not match actuals."""
        teams = ["A", "B"]
        position_odds = pd.DataFrame(
            [[0.5, 0.5], [0.5, 0.5]], index=teams, columns=[1, 2]
        )
        mean_ft = pd.DataFrame({"Team": teams, "Pos": [1, 2], "Points": [60., 50.]})
        actual = pd.DataFrame({"Team": teams, "Pos": [1, 2], "Points": [60., 50.]})
        ev = Evaluate(_make_sim(position_odds, mean_ft), actual)
        self.assertGreater(ev.get_brier_score(), 0.0)


class TestGetLogLoss(unittest.TestCase):
    def test_perfect_prediction_returns_zero(self):
        """Log loss is 0.0 when each team has probability 1.0 at their actual position."""
        sim, actual = _perfect_sim_and_actual()
        ev = Evaluate(sim, actual)
        self.assertAlmostEqual(ev.get_log_loss(), 0.0, places=10)

    def test_zero_probability_clipped_not_raised(self):
        """Log loss clips predicted probability 0 to 1e-15 and returns a finite value."""
        teams = ["A", "B"]
        # A predicted to finish 2nd (prob 0 at pos 1), B predicted to finish 1st (prob 0 at pos 2)
        position_odds = pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]], index=teams, columns=[1, 2]
        )
        mean_ft = pd.DataFrame({"Team": teams, "Pos": [1, 2], "Points": [60., 50.]})
        actual = pd.DataFrame({"Team": teams, "Pos": [1, 2], "Points": [60., 50.]})
        ev = Evaluate(_make_sim(position_odds, mean_ft), actual)
        result = ev.get_log_loss()
        self.assertTrue(math.isfinite(result))
        self.assertGreater(result, 0.0)


class TestGetSpearmansRankCoefficient(unittest.TestCase):
    def test_perfect_ranking_returns_one(self):
        """Spearman coefficient is 1.0 when predicted ranking matches actuals exactly."""
        sim, actual = _perfect_sim_and_actual()
        ev = Evaluate(sim, actual)
        self.assertAlmostEqual(ev.get_spearmans_rank_coefficient(), 1.0, places=10)

    def test_fully_reversed_ranking_returns_minus_one(self):
        """Spearman coefficient is -1.0 when predicted ranking is the exact reverse of actuals."""
        teams = ["A", "B", "C", "D"]
        position_odds = pd.DataFrame(np.eye(4), index=teams, columns=[1,2,3,4])
        mean_ft = pd.DataFrame({"Team": teams, "Pos": [4,3,2,1], "Points": [50.,60.,70.,80.]})
        actual = pd.DataFrame({"Team": teams, "Pos": [1,2,3,4], "Points": [80.,70.,60.,50.]})
        ev = Evaluate(_make_sim(position_odds, mean_ft), actual)
        self.assertAlmostEqual(ev.get_spearmans_rank_coefficient(), -1.0, places=10)


class TestGetKendallRankCoefficient(unittest.TestCase):
    def test_perfect_ranking_returns_one(self):
        """Kendall tau is 1.0 when predicted ranking matches actuals exactly."""
        sim, actual = _perfect_sim_and_actual()
        ev = Evaluate(sim, actual)
        self.assertAlmostEqual(ev.get_kendall_rank_coefficient(), 1.0, places=10)

    def test_fully_reversed_ranking_returns_minus_one(self):
        """Kendall tau is -1.0 when predicted ranking is the exact reverse of actuals."""
        teams = ["A", "B", "C", "D"]
        position_odds = pd.DataFrame(np.eye(4), index=teams, columns=[1,2,3,4])
        mean_ft = pd.DataFrame({"Team": teams, "Pos": [4,3,2,1], "Points": [50.,60.,70.,80.]})
        actual = pd.DataFrame({"Team": teams, "Pos": [1,2,3,4], "Points": [80.,70.,60.,50.]})
        ev = Evaluate(_make_sim(position_odds, mean_ft), actual)
        self.assertAlmostEqual(ev.get_kendall_rank_coefficient(), -1.0, places=10)

    def test_fewer_than_2_teams_returns_nan(self):
        """Kendall tau returns nan when the aligned dataset has fewer than 2 teams."""
        sim, actual = _perfect_sim_and_actual()
        ev = Evaluate(sim, actual)
        ev.aligned_positions_and_points["position_actual"] = np.array([1])
        ev.aligned_positions_and_points["position_prediction"] = np.array([1])
        self.assertTrue(math.isnan(ev.get_kendall_rank_coefficient()))


class TestPointsMetrics(unittest.TestCase):
    def setUp(self):
        self.sim, self.actual = _perfect_sim_and_actual()
        self.ev = Evaluate(self.sim, self.actual)

    def test_mae_returns_zero_for_perfect_predictions(self):
        """MAE is 0.0 when predicted points exactly match actual points."""
        self.assertAlmostEqual(self.ev.get_points_mae(), 0.0, places=10)

    def test_rmse_returns_zero_for_perfect_predictions(self):
        """RMSE is 0.0 when predicted points exactly match actual points."""
        self.assertAlmostEqual(self.ev.get_points_rmse(), 0.0, places=10)

    def test_mape_returns_zero_for_perfect_predictions(self):
        """MAPE is 0.0 when predicted points exactly match actual points."""
        self.assertAlmostEqual(self.ev.get_points_mape(), 0.0, places=10)

    def test_r2_returns_one_for_perfect_predictions(self):
        """R2 is 1.0 (not 0.0) when predicted points exactly match actual points."""
        self.assertAlmostEqual(self.ev.get_points_r2(), 1.0, places=10)

    def test_mae_positive_for_imperfect_predictions(self):
        """MAE is positive when predicted points deviate from actuals."""
        teams = ["A", "B", "C", "D"]
        position_odds = pd.DataFrame(np.eye(4), index=teams, columns=[1,2,3,4])
        mean_ft = pd.DataFrame({"Team": teams, "Pos": [1,2,3,4],
                                 "Points": [75.0, 65.0, 55.0, 45.0]})
        actual = pd.DataFrame({"Team": teams, "Pos": [1,2,3,4],
                                "Points": [80.0, 70.0, 60.0, 50.0]})
        ev = Evaluate(_make_sim(position_odds, mean_ft), actual)
        self.assertGreater(ev.get_points_mae(), 0.0)


class TestGetMetricsReport(unittest.TestCase):
    def setUp(self):
        sim, actual = _perfect_sim_and_actual()
        self.ev = Evaluate(sim, actual)

    def test_returns_dataframe(self):
        """get_metrics_report returns a pandas DataFrame."""
        self.assertIsInstance(self.ev.get_metrics_report(), pd.DataFrame)

    def test_has_correct_columns(self):
        """Returned DataFrame has exactly the columns Metric Group, Metric, Value."""
        report = self.ev.get_metrics_report()
        self.assertEqual(list(report.columns), ["Metric Group", "Metric", "Value"])

    def test_contains_all_expected_metrics(self):
        """Report includes every metric name from proper, ranking, and points groups."""
        expected = {"rps", "brier_score", "log_loss",
                    "spearmans_rank", "kendalls_rank",
                    "points_mae", "points_rmse", "points_mape", "points_r2"}
        report = self.ev.get_metrics_report()
        self.assertEqual(set(report["Metric"]), expected)

    def test_values_rounded_to_4_decimal_places(self):
        """Value column entries are rounded to 4 decimal places."""
        report = self.ev.get_metrics_report()
        for val in report["Value"]:
            val = float(val)
            if not math.isnan(val) and not math.isinf(val):
                with self.subTest(val=val):
                    self.assertAlmostEqual(val, round(val, 4), places=10)


class TestMetricsDict(unittest.TestCase):
    def setUp(self):
        sim, actual = _perfect_sim_and_actual()
        self.ev = Evaluate(sim, actual)

    def test_filters_to_requested_groups_only(self):
        """metrics_dict returns only the groups explicitly requested."""
        result = self.ev.metrics_dict(metrics_groups=("proper",))
        self.assertEqual(set(result.keys()), {"PROPER"})

    def test_all_three_groups_returned_by_default(self):
        """metrics_dict includes PROPER, RANKING, and POINTS when called with defaults."""
        result = self.ev.metrics_dict()
        self.assertEqual(set(result.keys()), {"PROPER", "RANKING", "POINTS"})


if __name__ == "__main__":
    unittest.main()
