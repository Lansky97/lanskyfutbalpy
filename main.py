from pprint import pprint
#from team import Team
from league import League 
from match import Match
from simulation import Simulation as Sim


match_data_location = 'data/result_24_25.csv'

league = League.from_matches(match_data_location, date_cutoff='2024-12-01', xG_factor=0.6)
simulations = Sim(league,2)
#match = Match(teams['Arsenal'], teams['Manchester City'], league.league_avarages[0], league.league_avarages[1])
#print(league.fixtures)
#pprint(league)
#print(league.league_table)
#pprint(league.teams)

pprint(simulations.simmed_leagues[0])
#print(simulations.simmed_leagues[1].league_table)

#print(match.match_expectation)
