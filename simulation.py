import copy
from match import Match, SimmedMatch

class Simulation:
    def __init__(self,league, n_trials=1000):
        self.league = league
        self.n_trials = n_trials
        self.simmed_leagues = self.simulate_season()

    def simulate_season(self):
        simmed_leagues = []

        for _ in range(self.n_trials):
            sim_league = copy.deepcopy(self.league)
            unique_dates = sim_league.fixtures['Date'].unique()
            for idx, match_date in enumerate(unique_dates):
                md_fixtures = sim_league.fixtures[sim_league.fixtures['Date'] == match_date]
                md_results = []
                day_matches = SimmedMatch.from_fixtures(sim_league.teams, md_fixtures, sim_league.xG_factor)
                md_results.extend([match.sim_result for match in day_matches])

                sim_league.update_league(md_results)

            sim_league.league_table = sim_league.generate_league_table()
            simmed_leagues.append(sim_league)
        return simmed_leagues