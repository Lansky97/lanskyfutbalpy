import copy
from match import Match, SimmedMatch
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from league import League
import time

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
        start_time = time.time()
        self.simmed_leagues: List[League] = self.simulate_season()
        self.run_time: float = time.time() - start_time
        self.mean_final_table: pd.DataFrame = self.mean_final_table()
        self.position_odds: pd.DataFrame = self.position_odds()
        self.competition_markets: pd.DataFrame = self.competition_markets()

    def simulate_season(self) -> List[League]:
        simmed_leagues: List[League] = []
        rng = np.random.default_rng(self.config.seed)

        fixtures_by_date: Dict[str, List[Dict[str, Any]]] = {}
        for fixture in self.league.fixtures:
            fixtures_by_date.setdefault(fixture['Date'], []).append(fixture)
        ordered_dates = sorted(fixtures_by_date.items())

        baseline_league = self._league_snapshot()

        for _ in range(self.n_trials):
            self._league_restore(baseline_league)

            for match_date, md_fixtures in ordered_dates:
                day_matches = SimmedMatch.from_fixtures(self.league.teams, md_fixtures, self.league.xG_factor, rng=rng)
                md_results = [match.sim_result for match in day_matches]
                self.league.update_league(md_results)

            trial = League(
                name=self.league.name,
                matches=self.league.matches,
                date_cutoff=self.league.date_cutoff_str,
                xG_factor=self.league.xG_factor,
                last_season_factor=self.league.last_season_factor
            )

            trial.teams = self.league.teams
            trial.results = self.league.results
            trial.league_table = self.league.generate_league_table()
            simmed_leagues.append(trial)

        return simmed_leagues

    def _league_snapshot(self) -> Dict[str, Any]:
        return {
            'teams': self._teams_snapshot(self.league.teams),
            'results_len': len(self.league.results)
        }

    def _league_restore(self, snapshot: Dict[str, Any]) -> None:
        self._restore_teams(self.league.teams, snapshot['teams'])
        # trim any appended results beyond baseline
        baseline_len = snapshot['results_len']
        if len(self.league.results) != baseline_len:
            del self.league.results[baseline_len:]

    def _teams_snapshot(self, teams: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        fields = (
            'home_games_played','away_games_played',
            'home_points','away_points',
            'home_goals','away_goals',
            'home_goals_against','away_goals_against',
            'home_xg','away_xg','home_xga','away_xga',
            'home_attack_strength_cs','away_attack_strength_cs',
            'home_defence_strength_cs','away_defence_strength_cs',
            'home_attack_strength','away_attack_strength',
            'home_defence_strength','away_defence_strength'
        )
        return {name: {field: getattr(team, field) for field in fields} for name, team in teams.items()}
    
    def _restore_teams(self, teams: Dict[str, Any], snapshot: Dict[str, Dict[str, Any]]) -> None:
        for name, state in snapshot.items():
            team = teams[name]
            for field, value in state.items():
                setattr(team, field, value)

    def mean_final_table(self):

        tables = [pd.DataFrame(sim.league_table) for sim in self.simmed_leagues]
        all_tables = pd.concat(tables, ignore_index=True)
        mean_table = all_tables.groupby('Team', as_index=False).mean()
        mean_table = mean_table.sort_values(by=(['Points', 'Goal Difference', 'Goals']), ascending=False)
        mean_table['Pos'] = range(1, len(mean_table) + 1) 
        return mean_table

    def position_odds(self):
        tables = [pd.DataFrame(sim.league_table) for sim in self.simmed_leagues]
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