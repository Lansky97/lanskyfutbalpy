import copy
from match import Match, SimmedMatch
import pandas as pd
import numpy as np
from typing import List, Optional
from league import League

class Config:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

class Simulation:
    def __init__(self, league: League, n_trials: int = 1000, config: Optional[Config] = None) -> None:
        if not isinstance(league, League):
            raise TypeError("league must be a League object.")
        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")
        self.league = league
        self.n_trials = n_trials
        self.config = config or Config()
        self.simmed_leagues: List[League] = self.simulate_season()
        self.mean_final_table: pd.DataFrame = self.mean_final_table()
        self.position_odds: pd.DataFrame = self.position_odds()
        self.competition_markets: pd.DataFrame = self.competition_markets()

    def simulate_season(self) -> List[League]:
        simmed_leagues: List[League] = []
        rng = np.random.default_rng(self.config.seed)
        for _ in range(self.n_trials):
            sim_league = copy.deepcopy(self.league)
            unique_dates = sim_league.fixtures['Date'].unique()
            for match_date in unique_dates:
                md_fixtures = sim_league.fixtures[sim_league.fixtures['Date'] == match_date]
                md_results = []
                day_matches = SimmedMatch.from_fixtures(sim_league.teams, md_fixtures, sim_league.xG_factor, rng=rng)
                md_results.extend([match.sim_result for match in day_matches])

                sim_league.update_league(md_results)

            sim_league.league_table = sim_league.generate_league_table()
            simmed_leagues.append(sim_league)
        return simmed_leagues

    def mean_final_table(self):
       
        tables = [sim.league_table for sim in self.simmed_leagues]
        all_tables = pd.concat(tables)
        mean_table = all_tables.groupby('Team', as_index=False).mean()
        mean_table = mean_table.sort_values(by=(['Points', 'Goal Difference', 'Goals']), ascending=False)
        mean_table['Pos'] = range(1, len(mean_table) + 1) 
        return mean_table

    def position_odds(self):
        tables = [sim.league_table for sim in self.simmed_leagues]
        all_tables = pd.concat(tables)
        all_tables['Pos'] = all_tables['Pos'].astype(int) 

        pos_counts = all_tables.groupby('Team')['Pos'].value_counts().unstack(fill_value=0)
        probs = pos_counts / self.n_trials

        probs['Avg_Pos'] = (probs * probs.columns).sum(axis=1)
        probs = probs.sort_values('Avg_Pos')
        probs = probs.drop(columns='Avg_Pos')
        return probs

    def competition_markets(self):
        probs = self.position_odds

        num_teams = probs.shape[1]

        markets = pd.DataFrame(index=probs.index)
        markets['Champion'] = probs[1]
        markets['Top 4'] = probs[[1,2,3,4]].sum(axis=1)
        markets['Top 7'] = probs[[1,2,3,4,5,6,7]].sum(axis=1)
        markets['Relegation'] = probs[[num_teams-2, num_teams-1, num_teams]].sum(axis=1)

        return markets