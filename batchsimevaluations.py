import pandas as pd 
from league import League
from simulation import Simulation, Config
from evaluate import Evaluate
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt

class BatchSimEvaluator:
    def __init__(self, input_csv: str):
       self.inputs_df = pd.read_csv(input_csv)
       self.league_table_cache: Dict[str, pd.DataFrame] = {}
       self.reports = self.run_evaluator()

    def _cache_key(self, params: Dict) -> str:
        return "|".join(f"{k}={params[k]}" for k in sorted(params.keys()))

    def get_final_league_table_csv(self, match_data_location: str, xG_factor: float) -> pd.DataFrame:
        params = {
            "mode": "csv",
            "match_data_location": match_data_location,
            "date_cutoff": "2100-01-01",
            "xG_factor": xG_factor
        }
        key = self._cache_key(params)
        if key not in self.league_table_cache:
            league = League.from_matches(match_data_location, params["date_cutoff"], xG_factor)
            self.league_table_cache[key] = league.league_table
        return pd.DataFrame(self.league_table_cache[key])
    
    def get_final_league_table_db(self, season_end_year: int, league: str, tier: int,
                                  xG_factor: float, last_season_factor: float) -> pd.DataFrame:
        params = {
            "mode": "db",
            "season_end_year": season_end_year,
            "league": league,
            "tier": tier,
            "date_cutoff": "2100-01-01",
            "xG_factor": xG_factor,
            "last_season_factor": last_season_factor
        }
        key = self._cache_key(params)
        if key not in self.league_table_cache:
            lge = League.from_database(
                season_end_year=season_end_year,
                league=league,
                tier=tier,
                date_cutoff=params["date_cutoff"],
                xG_factor=xG_factor,
                last_season_factor=last_season_factor
            )
            self.league_table_cache[key] = lge.league_table
        return pd.DataFrame(self.league_table_cache[key])
    
    def run_evaluator(self) -> pd.DataFrame:
        reports = []
        sim_count = 0
        for _, row in self.inputs_df.iterrows():
            is_db = {'season_end_year', 'league', 'tier'}.issubset(row.index)
            xG_factor = float(row['xG_factor'])
            date_cutoff = str(row['date_cutoff'])
            n_trials = int(row['n_trials'])
            seed = int(row['seed'])
            if is_db:
                season_end_year = int(row['season_end_year'])
                league_name = str(row['league'])
                tier = int(row['tier'])
                last_season_factor = float(row['last_season_factor'])
                actual_final_table = self.get_final_league_table_db(
                    season_end_year, league_name, tier, xG_factor, last_season_factor
                )
                league = League.from_database(
                    season_end_year, league_name, tier, date_cutoff, xG_factor, last_season_factor
                )
            else:
                match_data_location = str(row['match_data_location'])
                actual_final_table = self.get_final_league_table_csv(
                    match_data_location, xG_factor
                )
                league = League.from_matches(match_data_location, date_cutoff, xG_factor)

            config = Config(seed=seed)
            sim_count += 1
            print(f"Running simulation {sim_count}")
            sim = Simulation(league, n_trials=n_trials, config=config)

            evaluation = Evaluate(simulation=sim, actual_final_table=actual_final_table)
            metrics_report = evaluation.metrics_report.copy()

            for column in self.inputs_df.columns:
                metrics_report[column] = row[column]

            metrics_report['run_time'] = sim.run_time
            reports.append(metrics_report)

        return pd.concat(reports, ignore_index=True)

    def plot_metrics(self, metric_group:str, tested_variable: str, group_by: str = None):
        plot_df = self.reports[self.reports['Metric Group'].str.upper() == metric_group.upper()]
        if plot_df.empty:
            raise ValueError(f"No data available for metric group: '{metric_group}'")

        group_metrics = plot_df['Metric'].unique()
        plot_df = plot_df[plot_df['Metric'].isin(group_metrics)]
        grid = sns.FacetGrid(plot_df, col='Metric', col_wrap=4, sharey=False, height=4)

        if group_by:
            grid.map_dataframe(sns.lineplot, x=tested_variable, y='Value', hue=group_by, 
                               marker='o', palette='tab10')
            grid.add_legend(title=group_by)
        else:
            grid.map_dataframe(sns.lineplot, x=tested_variable, y='Value', color='red', 
                               linestyle=':', marker='o', markerfacecolor='black')

        grid.set_titles(col_template="{col_name}")
        grid.set_axis_labels(x_var=tested_variable, y_var='Value')

        for ax in grid.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

        plt.tight_layout()
        title_suffix = f" grouped by {group_by}" if group_by else ""
        grid.fig.suptitle(f"Metrics for {metric_group} by {tested_variable}{title_suffix}")
        plt.show()

    def plot_run_time(self, tested_variable: str):
 
        df = self.reports.drop_duplicates(subset=[tested_variable, 'run_time'])
        plt.figure(figsize=(7, 5))
        sns.lineplot(data=df, x=tested_variable, y='run_time', color='red', linestyle=':', marker='o', markerfacecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(tested_variable)
        plt.ylabel('Simulation Run Time (seconds)')
        plt.title(f'Simulation Run Time vs {tested_variable}')
        plt.tight_layout()
        plt.show()

