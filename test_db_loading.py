import unittest
from league import League
from team import Team
import pandas as pd

class TestDatabaseLoading(unittest.TestCase):
    def test_from_database_defaults(self):
        # Test loading with defaults (Premier League 2024)
        # This assumes the database is present and populated as per previous exploration
        try:
            league = League.from_database(season_end_year=2025, league='Premier_League', tier=1)
        except Exception as e:
            self.fail(f"League.from_database raised exception: {e}")

        self.assertIsInstance(league, League)
        self.assertGreater(len(league.teams), 0)
        self.assertGreater(len(league.results), 0)
        
        # Check if a known team has historical stats
        # Assuming 'Arsenal' or similar exists and had stats last season
        first_team = list(league.teams.values())[0]
        # We can't be sure which team is first, but at least one team should have non-zero LS stats if data exists
        
        has_ls_stats = any(t.home_attack_strength_ls != 0 for t in league.teams.values())
        self.assertTrue(has_ls_stats, "No teams have historical (LS) stats populated")

    def test_from_database_championship(self):
        # Test loading a different league/tier
        try:
            league = League.from_database(season_end_year=2025, league='Championship', tier=2)
        except Exception as e:
            self.fail(f"League.from_database for Championship raised exception: {e}")
            
        self.assertGreater(len(league.teams), 0)

    def test_factor_zero(self):
        # Test that factor=0 results in 0 strength for teams with no games (if any) or ignores LS stats
        league = League.from_database(season_end_year=2025, league='Premier_League', tier=1, last_season_factor=0.0)
        
        # If a team has 0 games played, its strength should be 0 (because factor is 0)
        # We can't guarantee a team has 0 games, but we can check that strength is NOT equal to LS strength if LS strength is non-zero
        # actually, if CS is 0, strength should be 0.
        
        for team in league.teams.values():
            if team.home_games_played == 0:
                self.assertEqual(team.home_attack_strength, 0.0, f"Team {team.name} has 0 games but non-zero strength with factor=0")

if __name__ == '__main__':
    unittest.main()
