from pprint import pprint
#from team import Team
from league import League 
from match import Match



match_data_location = 'data/result_24_25.csv'

league = League.from_matches(match_data_location, date_cutoff='2024-12-01', xG_factor=0.6)
#match = Match(teams['Arsenal'], teams['Manchester City'], league.league_avarages[0], league.league_avarages[1])
#print(league.fixtures)
#pprint(league)
#print(league.league_table)
pprint(league.teams)

#print(match.match_expectation)
