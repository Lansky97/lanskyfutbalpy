from pprint import pprint
from league import League 
from simulation import Simulation as Sim
from simulation import Config 


xG_factor = 0.6
date_cutoff = '2025-01-01'
config = Config(419)
#match_data_location = 'data/result_24_25.csv'

league = League.from_database(season_end_year=2025, league='Premier_League', tier=1, date_cutoff=date_cutoff, xG_factor=xG_factor, last_season_factor= 0.1)
simulations = Sim(
    league,
    n_trials=1000,
    config=config
)
#match = Match(teams['Arsenal'], teams['Manchester City'], league.league_avarages[0], league.league_avarages[1])
#print(league.fixtures)
#pprint(league)
#print(league.league_table)
#pprint(league.teams)

#pprint(simulations.simmed_leagues[0])
#print(simulations.mean_final_table)

#print(match.match_expectation)
