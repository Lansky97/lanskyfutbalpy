from simulation import Simulation
import pandas as pd
import numpy as np

class Evaluate:
    def __init__(self,
        simulation: Simulation,
        actual_final_table: pd.DataFrame = None
    ):
        self.simulation = simulation
        self.actual_final_table = actual_final_table 
        self.probs = self._prep_probs()
        self.actuals = self._prep_actuals_matrix()
        self.aligned_positions_and_points = self._prep_team_positions_and_points()
        self.metrics_report = self.get_metrics_report()

    def metrics_dict(self, metrics_groups = ("proper", "ranking", "points")) -> dict:
       metrics = {
          "proper": self.proper_score_metrics(),
          "ranking": self.ranking_metrics(),
          "points": self.points_metrics(),
       }
       output = {}
       for metric_group in metrics_groups:
           output[metric_group.upper()] = metrics[metric_group]
       return output

    def get_metrics_report(self, metrics_groups = ("proper", "ranking", "points"), round_digits: int = 4) -> pd.DataFrame:
       nested = self.metrics_dict(metrics_groups)
       rows = []
       for metric_group, metric_dict in nested.items():
           for metric, value in metric_dict.items():
               rows.append((metric_group, metric, value))
       
       output = pd.DataFrame(rows, columns=["Metric Group", "Metric", "Value"])
       output["Value"] = output["Value"].round(round_digits)

       order = {
        "PROPER":      ["rps", "brier_score", "log_loss"],
        "RANKING":    ["spearmans_rank", "kendalls_rank"],
        "POINTS": ["points_mae", "points_rmse", "points_mape", "points_r2"]
       }

       return output.sort_values(["Metric Group", "Metric"]).reset_index(drop=True)
    
    def proper_score_metrics(self) -> dict:
       return {
           "rps": self.get_ranked_probability_score(),
           "brier_score": self.get_brier_score(),
           "log_loss": self.get_log_loss()
       }
    
    def ranking_metrics(self) -> dict:
       return {
           "spearmans_rank": self.get_spearmans_rank_coefficient(),
           "kendalls_rank": self.get_kendall_rank_coefficient()
       }

    def points_metrics(self) -> dict:
       return {
           "points_mae": self.get_points_mae(),
           "points_rmse": self.get_points_rmse(),
           "points_mape": self.get_points_mape(),
           "points_r2": self.get_points_r2()
       }

    def get_ranked_probability_score(self) -> float:
        probs_df = self.probs['probs_df']
        num_positions = self.probs['num_positions']

        cum_probs_df = np.cumsum(probs_df, axis=1)[:, :-1]
        cum_actuals = self.actuals['cumulative_actuals']

        rps_per_team = np.sum((cum_probs_df - cum_actuals) ** 2, axis=1) / (num_positions - 1)
        return float(np.mean(rps_per_team))
    
    def get_brier_score(self) -> float:
        probs_df = self.probs['probs_df']
        actuals = self.actuals['one_hot_actuals']
        brier_per_team = np.mean((probs_df - actuals) ** 2, axis=1)

        return float(np.mean(brier_per_team))

    def get_log_loss(self) -> float:
        probs_df = self.probs['probs_df']
        actual_pos = self.probs['actual_pos']
        num_positions = self.probs['num_positions']

        pred_prob = probs_df[np.arange(num_positions), actual_pos - 1]
        pred_prob = np.clip(pred_prob, 1e-15, 1)
        return float(np.mean(-np.log(pred_prob)))

    def get_spearmans_rank_coefficient(self) -> float:
        actual_pos = self.aligned_positions_and_points['position_actual']
        pred_pos = self.aligned_positions_and_points['position_prediction']

        return float(np.corrcoef(actual_pos, pred_pos)[0, 1])

    def get_kendall_rank_coefficient(self) -> float:
        actual_pos = self.aligned_positions_and_points['position_actual']
        pred_pos = self.aligned_positions_and_points['position_prediction']

        n = actual_pos.shape[0]
        if n < 2:
            return float('nan')
        
        diff_actuals = actual_pos[:, None] - actual_pos[None, :]
        diff_preds = pred_pos[:, None] - pred_pos[None, :]
        comparison_matrix = np.sign(diff_actuals) * np.sign(diff_preds)

        upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)
        comparison_matrix_upper = comparison_matrix[upper_triangle]
        concordant = np.count_nonzero(comparison_matrix_upper > 0)
        discordant = np.count_nonzero(comparison_matrix_upper < 0)
        denominator = n * (n - 1) / 2
        tau_a = (concordant - discordant) / denominator
        return float(tau_a)

    def get_points_mae(self) -> float:
        actuals = self.aligned_positions_and_points['points_actual']
        preds = self.aligned_positions_and_points['points_prediction']
        return float(np.mean(np.abs(actuals - preds)))

    def get_points_rmse(self) -> float:
        actuals = self.aligned_positions_and_points['points_actual']
        preds = self.aligned_positions_and_points['points_prediction']
        return float(np.sqrt(np.mean((actuals - preds) ** 2)))

    def get_points_mape(self) -> float:
        actuals = self.aligned_positions_and_points['points_actual']
        preds = self.aligned_positions_and_points['points_prediction']

        return float(np.mean(np.abs((actuals - preds) / actuals)))

    def get_points_r2(self) -> float:
        actuals = self.aligned_positions_and_points['points_actual']
        preds = self.aligned_positions_and_points['points_prediction']
        actual_mean = np.mean(actuals)
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - actual_mean) ** 2)
        return float(1 - (ss_res / ss_tot))

    def _prep_probs(self) -> dict:
        probs = self.simulation.position_odds
        actual = self.actual_final_table

        actual_pos = actual['Pos'].astype(int).to_numpy()
        teams = actual['Team']
        num_positions = len(actual)

        expected_cols = list(range(1, num_positions + 1))
        probs_df = probs.reindex(index=teams, columns=expected_cols, fill_value=0)

        probs_df = probs_df.to_numpy(dtype=float)
        return {
            'probs_df': probs_df,
            'actual_pos': actual_pos,
            'num_positions': num_positions
        }

    def _prep_actuals_matrix(self) -> dict:
        actual_pos = self.probs['actual_pos']
        num_positions = self.probs['num_positions']

        one_hot_actuals = np.zeros((num_positions, num_positions), dtype=float)
        one_hot_actuals[np.arange(num_positions), actual_pos - 1] = 1

        grid = np.arange(num_positions-1)[None, :]
        threshold = (actual_pos[:, None] - 1)
        cumulative_actuals = (grid >= threshold).astype(float)

        return {
            'one_hot_actuals': one_hot_actuals,
            'cumulative_actuals': cumulative_actuals
        }

    def _prep_team_positions_and_points(self) -> dict:
       actuals = self.actual_final_table[['Team', 'Pos', 'Points']].copy()
       preds = self.simulation.mean_final_table[['Team', 'Pos', 'Points']].copy()

       actuals['Pos'] = actuals['Pos'].astype(int)
       preds['Pos'] = preds['Pos'].astype(int)

       actuals['Points'] = actuals['Points'].astype(float)
       preds['Points'] = preds['Points'].astype(float)

       merged = actuals.merge(preds, on='Team', how='inner', suffixes=('_actual', '_pred'))
       pos_actual = merged['Pos_actual'].to_numpy(dtype=int)
       points_actual = merged['Points_actual'].to_numpy(dtype=float)
       pos_pred = merged['Pos_pred'].to_numpy(dtype=int)
       points_pred = merged['Points_pred'].to_numpy(dtype=float)

       return {
           'position_actual': pos_actual,
           'points_actual': points_actual,
           'position_prediction': pos_pred,
           'points_prediction': points_pred
       }
