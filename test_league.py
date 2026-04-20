import unittest
from unittest.mock import patch, MagicMock
from league import League
from team import Team


# ─── shared helpers ─────────────────────────────────────────────────────────

def _full_row(date, home='A', away='B', hg=1, ag=1, hxg=1.0, axg=1.0):
    """A match row with all required keys."""
    return {
        'Date': date, 'Home': home, 'Away': away,
        'HomeGoals': hg, 'AwayGoals': ag,
        'Home_xG': hxg, 'Away_xG': axg,
        'Competition_Name': 'Test League',
    }


def _sparse_row(date, home='A', away='B'):
    """A match row with only Date/Home/Away and sentinel None goals (postponed-style)."""
    return {
        'Date': date, 'Home': home, 'Away': away,
        'HomeGoals': None, 'AwayGoals': None,
        'Home_xG': None, 'Away_xG': None,
    }


# ─── __init__ ────────────────────────────────────────────────────────────────

class TestLeagueInit(unittest.TestCase):

    def _two_results(self):
        return [
            _full_row('2024-08-01', 'A', 'B', 2, 1, 1.5, 1.2),
            _full_row('2024-08-02', 'B', 'A', 0, 2, 0.8, 1.7),
        ]

    def test_valid_construction_stores_attributes(self):
        """Valid arguments produce a League with the correct name, xG_factor, and cutoff."""
        lge = League('PL', self._two_results(), '2024-12-01', xG_factor=0.6)
        self.assertEqual(lge.name, 'PL')
        self.assertEqual(lge.xG_factor, 0.6)
        self.assertEqual(lge.date_cutoff_str, '2024-12-01')
        self.assertIsInstance(lge.results, list)
        self.assertIsInstance(lge.fixtures, list)

    def test_invalid_name_type_raises_type_error(self):
        """A non-string name raises TypeError."""
        with self.assertRaises(TypeError):
            League(123, [], '2024-12-01', xG_factor=0.6)

    def test_invalid_matches_not_list_raises_type_error(self):
        """Passing a non-list for matches raises TypeError."""
        with self.assertRaises(TypeError):
            League('PL', 'not a list', '2024-12-01', xG_factor=0.6)

    def test_invalid_matches_list_of_non_dicts_raises_type_error(self):
        """A list whose elements are not dicts raises TypeError."""
        with self.assertRaises(TypeError):
            League('PL', ['not', 'dicts'], '2024-12-01', xG_factor=0.6)

    def test_xg_factor_int_is_accepted(self):
        """An integer xG_factor (e.g. 1) is accepted now that the check allows int."""
        lge = League('PL', [], '2024-12-01', xG_factor=1)
        self.assertEqual(lge.xG_factor, 1)

    def test_invalid_xg_factor_string_raises_type_error(self):
        """Passing xG_factor as a string raises TypeError."""
        with self.assertRaises(TypeError):
            League('PL', [], '2024-12-01', xG_factor='0.6')

    def test_bad_date_cutoff_format_raises_value_error(self):
        """A date_cutoff in DD-MM-YYYY format raises ValueError."""
        with self.assertRaises(ValueError):
            League('PL', [], '01-12-2024', xG_factor=0.6)

    def test_non_string_date_cutoff_raises_value_error(self):
        """A non-string date_cutoff such as None raises ValueError."""
        with self.assertRaises(ValueError):
            League('PL', [], None, xG_factor=0.6)

    def test_teams_initialised_empty(self):
        """League.teams starts as an empty dict; population is the caller's responsibility."""
        lge = League('PL', [], '2024-12-31', xG_factor=0.6)
        self.assertEqual(lge.teams, {})

    def test_league_table_initialised_empty(self):
        """League.league_table starts as an empty list."""
        lge = League('PL', [], '2024-12-31', xG_factor=0.6)
        self.assertEqual(lge.league_table, [])


# ─── generate_results ────────────────────────────────────────────────────────

class TestGenerateResults(unittest.TestCase):

    def test_includes_row_on_cutoff_date(self):
        """A row whose Date equals date_cutoff (≤) is included in results."""
        matches = [_full_row('2024-12-01')]
        lge = League('T', matches, '2024-12-01', xG_factor=0.6)
        self.assertEqual(len(lge.results), 1)
        self.assertEqual(lge.results[0]['Date'], '2024-12-01')

    def test_excludes_future_matches(self):
        """Rows with Date > date_cutoff are excluded from results."""
        matches = [_full_row('2025-01-01')]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        self.assertEqual(lge.results, [])

    def test_filters_by_date_cutoff(self):
        """Only rows with Date <= date_cutoff appear in results; future rows are absent."""
        matches = [
            _full_row('2024-08-01'),   # before cutoff → result
            _full_row('2024-12-01'),   # on cutoff → result
            _full_row('2024-12-02'),   # after cutoff → fixture only
        ]
        lge = League('T', matches, '2024-12-01', xG_factor=0.6)
        result_dates = {r['Date'] for r in lge.results}
        self.assertIn('2024-08-01', result_dates)
        self.assertIn('2024-12-01', result_dates)
        self.assertNotIn('2024-12-02', result_dates)

    def test_skips_row_with_none_goals(self):
        """Rows with HomeGoals=None are silently skipped (postponed match handling)."""
        matches = [
            _full_row('2024-08-01', hg=None),   # skipped
            _full_row('2024-08-02', hg=2, ag=0),  # kept
        ]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        self.assertEqual(len(lge.results), 1)
        self.assertEqual(lge.results[0]['Date'], '2024-08-02')

    def test_skips_row_with_non_numeric_goals(self):
        """Rows where HomeGoals is a non-numeric string are silently skipped."""
        matches = [_full_row('2024-08-01', hg='N/A')]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        self.assertEqual(lge.results, [])

    def test_raises_on_missing_required_key(self):
        """A result row missing a required key raises ValueError naming the key."""
        bad_row = {
            'Date': '2024-08-01', 'Away': 'B',   # 'Home' missing
            'HomeGoals': 1, 'AwayGoals': 1,
            'Home_xG': 1.0, 'Away_xG': 1.0,
        }
        with self.assertRaises(ValueError) as ctx:
            League('T', [bad_row], '2024-12-31', xG_factor=0.6)
        self.assertIn('Home', str(ctx.exception))

    def test_pts_computed_correctly(self):
        """Home win produces Home_pts=3, Away_pts=0; draw produces 1, 1."""
        matches = [
            _full_row('2024-08-01', hg=2, ag=0),  # home win
            _full_row('2024-08-02', hg=1, ag=1),  # draw
        ]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        by_date = {r['Date']: r for r in lge.results}
        self.assertEqual(by_date['2024-08-01']['Home_pts'], 3)
        self.assertEqual(by_date['2024-08-01']['Away_pts'], 0)
        self.assertEqual(by_date['2024-08-02']['Home_pts'], 1)
        self.assertEqual(by_date['2024-08-02']['Away_pts'], 1)

    def test_results_sorted_ascending_by_date(self):
        """Results are returned in ascending date order regardless of input order."""
        matches = [
            _full_row('2024-08-10'),
            _full_row('2024-08-01'),
            _full_row('2024-08-05'),
        ]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        dates = [r['Date'] for r in lge.results]
        self.assertEqual(dates, sorted(dates))


# ─── generate_fixtures ───────────────────────────────────────────────────────

class TestGenerateFixtures(unittest.TestCase):
    # Sparse rows (None goals) are used for past/cutoff rows so generate_results skips
    # them gracefully rather than raising on missing goal data.

    def test_excludes_past_and_cutoff_matches(self):
        """Rows with Date <= date_cutoff do not appear in fixtures."""
        matches = [
            _sparse_row('2024-08-01'),   # past → not a fixture
            _sparse_row('2024-12-01'),   # on cutoff → not a fixture
            _sparse_row('2024-12-02'),   # future → fixture
        ]
        lge = League('T', matches, '2024-12-01', xG_factor=0.6)
        fixture_dates = {f['Date'] for f in lge.fixtures}
        self.assertNotIn('2024-08-01', fixture_dates)
        self.assertNotIn('2024-12-01', fixture_dates)
        self.assertIn('2024-12-02', fixture_dates)

    def test_deduplicates_identical_fixtures(self):
        """Two rows with the same (Date, Home, Away) produce only one fixture."""
        matches = [
            _sparse_row('2025-01-01', 'A', 'B'),
            _sparse_row('2025-01-01', 'A', 'B'),  # duplicate
        ]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        self.assertEqual(len(lge.fixtures), 1)

    def test_fixtures_sorted_ascending_by_date(self):
        """Fixtures are returned in ascending date order."""
        matches = [
            _sparse_row('2025-03-01'),
            _sparse_row('2025-01-01'),
            _sparse_row('2025-02-01'),
        ]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        dates = [f['Date'] for f in lge.fixtures]
        self.assertEqual(dates, sorted(dates))

    def test_fixture_output_contains_only_date_home_away(self):
        """Each fixture dict contains exactly the keys Date, Home, and Away."""
        matches = [_sparse_row('2025-01-01')]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        self.assertEqual(set(lge.fixtures[0].keys()), {'Date', 'Home', 'Away'})

    def test_different_team_pairings_not_deduplicated(self):
        """Two fixtures with different teams on the same date both appear."""
        matches = [
            _sparse_row('2025-01-01', 'A', 'B'),
            _sparse_row('2025-01-01', 'C', 'D'),
        ]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        self.assertEqual(len(lge.fixtures), 2)


# ─── league averages ─────────────────────────────────────────────────────────

class TestLeagueAverages(unittest.TestCase):
    # Two result rows (xG_factor=0.6):
    # total_home_goals=2, total_away_goals=3
    # total_home_xg=2.3, total_away_xg=2.9
    # smooth_home = 0.4*2 + 0.6*2.3 = 2.18  → avg = 2.18/2 = 1.09
    # smooth_away = 0.4*3 + 0.6*2.9 = 2.94  → avg = 2.94/2 = 1.47

    def _two_results(self):
        return [
            _full_row('2024-08-01', 'A', 'B', hg=2, ag=1, hxg=1.5, axg=1.2),
            _full_row('2024-08-02', 'B', 'A', hg=0, ag=2, hxg=0.8, axg=1.7),
        ]

    def test_avg_calculated_correctly_from_results(self):
        """league_avg_home and league_avg_away match the smoothed-goals formula."""
        lge = League('T', self._two_results(), '2024-12-31', xG_factor=0.6)
        self.assertAlmostEqual(lge.league_avg_home, 1.09, places=9)
        self.assertAlmostEqual(lge.league_avg_away, 1.47, places=9)

    def test_avg_defaults_to_1_when_no_results(self):
        """With no results (all matches in the future), both averages default to 1.0."""
        # Include all required keys with None goals — generate_results now validates
        # every row before the date filter, then skips on TypeError from int(None).
        matches = [{'Date': '2099-01-01', 'Home': 'A', 'Away': 'B',
                    'HomeGoals': None, 'AwayGoals': None, 'Home_xG': None, 'Away_xG': None}]
        lge = League('T', matches, '2024-12-31', xG_factor=0.6)
        self.assertEqual(lge.league_avg_home, 1.0)
        self.assertEqual(lge.league_avg_away, 1.0)

    def test_games_played_reflects_result_count(self):
        """games_played equals the number of rows returned by generate_results."""
        lge = League('T', self._two_results(), '2024-12-31', xG_factor=0.6)
        self.assertEqual(lge.games_played, 2)

    def test_xg_factor_blends_goals_and_xg(self):
        """A higher xG_factor shifts the average closer to the xG values."""
        # One result: goals=0, xg=2.0 (goals and xG differ maximally)
        matches = [_full_row('2024-08-01', hg=0, ag=0, hxg=2.0, axg=2.0)]
        lge_low = League('T', matches, '2024-12-31', xG_factor=0.0)   # pure goals
        lge_high = League('T', matches, '2024-12-31', xG_factor=1.0)  # pure xG
        self.assertAlmostEqual(lge_low.league_avg_home, 0.0)
        self.assertAlmostEqual(lge_high.league_avg_home, 2.0)


# ─── generate_league_table ───────────────────────────────────────────────────

class TestGenerateLeagueTable(unittest.TestCase):

    def _make_team(self, name, pts, goals, goals_against):
        """Create a Team with home-only stats for simplicity."""
        t = Team(name)
        t.home_games_played = 2
        t.home_points = pts
        t.home_goals = goals
        t.home_goals_against = goals_against
        return t

    def setUp(self):
        self.lge = League('T', [], '2024-12-31', xG_factor=0.6)
        # A: 6 pts, GD=+3, 4 goals  → Pos 1 (most points)
        # B: 3 pts, GD=+1, 3 goals  → Pos 2 (beats C on goals)
        # C: 3 pts, GD=+1, 2 goals  → Pos 3
        self.lge.teams = {
            'A': self._make_team('A', pts=6, goals=4, goals_against=1),
            'B': self._make_team('B', pts=3, goals=3, goals_against=2),
            'C': self._make_team('C', pts=3, goals=2, goals_against=1),
        }

    def test_table_has_required_columns(self):
        """Each table row has the expected set of keys including Pos."""
        table = self.lge.generate_league_table()
        expected_keys = {'Team', 'Played', 'Points', 'Goals', 'Goals Against',
                         'Goal Difference', 'xG', 'xGA', 'Pos'}
        self.assertEqual(set(table[0].keys()), expected_keys)

    def test_table_length_equals_team_count(self):
        """The table has exactly as many rows as there are teams."""
        table = self.lge.generate_league_table()
        self.assertEqual(len(table), 3)

    def test_sorted_by_points_descending(self):
        """The team with the most points occupies Pos 1."""
        table = self.lge.generate_league_table()
        self.assertEqual(table[0]['Team'], 'A')

    def test_sorted_by_goals_when_points_and_gd_tied(self):
        """When points and goal difference are equal, higher goals scores rank higher."""
        table = self.lge.generate_league_table()
        names = [row['Team'] for row in table]
        self.assertLess(names.index('B'), names.index('C'))

    def test_pos_assigned_sequentially_from_one(self):
        """Pos is 1 for the leader and increments by 1 for each subsequent team."""
        table = self.lge.generate_league_table()
        self.assertEqual([row['Pos'] for row in table], [1, 2, 3])

    def test_goal_difference_computed_correctly(self):
        """Goal Difference equals Goals minus Goals Against for every row."""
        table = self.lge.generate_league_table()
        for row in table:
            self.assertEqual(row['Goal Difference'], row['Goals'] - row['Goals Against'])

    def test_empty_teams_returns_empty_table(self):
        """With no teams in the league, generate_league_table returns an empty list."""
        lge = League('T', [], '2024-12-31', xG_factor=0.6)
        self.assertEqual(lge.generate_league_table(), [])


# ─── update_league ───────────────────────────────────────────────────────────

class TestUpdateLeague(unittest.TestCase):
    # initial: 1 home result (draw 1-1, Home_xG=1.2, Away_xG=1.0)
    # smooth_home=0.4*1+0.6*1.2=1.12 → league_avg_home=1.12

    def setUp(self):
        self.initial = [_full_row('2024-08-01', 'A', 'B', hg=1, ag=1, hxg=1.2, axg=1.0)]
        self.lge = League('T', self.initial, '2024-12-31', xG_factor=0.6)
        self.lge.teams = Team.teams_from_results(
            self.lge.results, self.lge.league_avg_home, self.lge.league_avg_away
        )

    def _new_result(self):
        return {
            'Date': '2024-08-08', 'Home': 'A', 'Away': 'B',
            'HomeGoals': 2, 'AwayGoals': 0,
            'Home_xG': 1.8, 'Away_xG': 0.5,
            'Home_pts': 3, 'Away_pts': 0,
        }

    def test_results_extended_after_update(self):
        """update_league appends new rows to self.results."""
        before = len(self.lge.results)
        self.lge.update_league([self._new_result()])
        self.assertEqual(len(self.lge.results), before + 1)

    def test_games_played_increments(self):
        """games_played increases by the number of new result rows."""
        before = self.lge.games_played
        self.lge.update_league([self._new_result()])
        self.assertEqual(self.lge.games_played, before + 1)

    def test_total_home_goals_accumulates(self):
        """total_home_goals grows by the HomeGoals in the new results."""
        before = self.lge.total_home_goals
        self.lge.update_league([self._new_result()])
        self.assertEqual(self.lge.total_home_goals, before + 2)

    def test_league_avg_recalculated_after_update(self):
        """league_avg_home changes after update because new totals are folded in."""
        avg_before = self.lge.league_avg_home
        self.lge.update_league([self._new_result()])
        # Initial avg = 1.12; new result (hg=2, hxg=1.8) raises the combined average
        self.assertNotAlmostEqual(self.lge.league_avg_home, avg_before, places=6)

    def test_multiple_new_results_all_processed(self):
        """Passing two new results increments games_played by 2."""
        second = dict(self._new_result())
        second['Date'] = '2024-08-15'
        self.lge.update_league([self._new_result(), second])
        self.assertEqual(self.lge.games_played, 3)


# ─── from_matches ────────────────────────────────────────────────────────────

class TestFromMatches(unittest.TestCase):

    def _match_rows(self, with_name=True):
        row = _full_row('2024-08-01', 'A', 'B')
        if with_name:
            row['Competition_Name'] = 'Premier League'
        else:
            row.pop('Competition_Name', None)
        return [row]

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_schedule', return_value=[])
    def test_empty_schedule_uses_unknown_league_name(self, mock_rs, mock_tfr):
        """When read_schedule returns an empty list, league_name falls back to 'Unknown League'."""
        lge = League.from_matches()
        self.assertEqual(lge.name, 'Unknown League')

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_schedule')
    def test_uses_competition_name_from_first_row(self, mock_rs, mock_tfr):
        """league_name is taken from Competition_Name in the first match row."""
        mock_rs.return_value = self._match_rows(with_name=True)
        lge = League.from_matches(date_cutoff='2024-12-31')
        self.assertEqual(lge.name, 'Premier League')

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_schedule')
    def test_missing_competition_name_falls_back_to_unknown(self, mock_rs, mock_tfr):
        """If Competition_Name is absent from match rows, name defaults to 'Unknown League'."""
        mock_rs.return_value = self._match_rows(with_name=False)
        lge = League.from_matches(date_cutoff='2024-12-31')
        self.assertEqual(lge.name, 'Unknown League')

    @patch('league.Team.teams_from_results')
    @patch('league.read_schedule')
    def test_teams_assigned_from_teams_from_results(self, mock_rs, mock_tfr):
        """League.teams is the return value of Team.teams_from_results."""
        mock_rs.return_value = self._match_rows()
        fake_teams = {'A': Team('A'), 'B': Team('B')}
        mock_tfr.return_value = fake_teams
        lge = League.from_matches(date_cutoff='2024-12-31')
        self.assertEqual(lge.teams, fake_teams)

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_schedule', return_value=[])
    def test_read_schedule_called_with_filepath(self, mock_rs, mock_tfr):
        """read_schedule is called exactly once with the filepath argument."""
        League.from_matches(match_data='fake/path.csv')
        mock_rs.assert_called_once_with(filepath='fake/path.csv')


# ─── from_database ───────────────────────────────────────────────────────────

class TestFromDatabase(unittest.TestCase):

    def _match_rows(self):
        row = _full_row('2024-08-01', 'A', 'B')
        row['Competition_Name'] = 'Premier League'
        row['Country'] = 'England'
        return [row]

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_last_season_stats', return_value=[])
    @patch('league.read_schedule')
    def test_read_last_season_stats_called_when_lsf_set(self, mock_rs, mock_rlss, mock_tfr):
        """read_last_season_stats is called when last_season_factor is not None."""
        mock_rs.return_value = self._match_rows()
        League.from_database(last_season_factor=0.5)
        mock_rlss.assert_called_once()

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_last_season_stats')
    @patch('league.read_schedule')
    def test_skips_read_last_season_stats_when_lsf_none(self, mock_rs, mock_rlss, mock_tfr):
        """read_last_season_stats is NOT called when last_season_factor is None."""
        mock_rs.return_value = self._match_rows()
        League.from_database(last_season_factor=None)
        mock_rlss.assert_not_called()

    @patch('league.Team.teams_from_results')
    @patch('league.read_last_season_stats', return_value=[])
    @patch('league.read_schedule')
    def test_teams_assigned_from_teams_from_results(self, mock_rs, mock_rlss, mock_tfr):
        """League.teams is the return value of Team.teams_from_results."""
        mock_rs.return_value = self._match_rows()
        fake_teams = {'A': MagicMock()}
        mock_tfr.return_value = fake_teams
        lge = League.from_database()
        self.assertEqual(lge.teams, fake_teams)

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_last_season_stats')
    @patch('league.read_schedule')
    def test_last_season_stats_stored_on_league(self, mock_rs, mock_rlss, mock_tfr):
        """The list returned by read_last_season_stats is stored as lge.last_season_stats."""
        mock_rs.return_value = self._match_rows()
        fake_stats = [{'team': 'A', 'home_attack_strength': 1.1}]
        mock_rlss.return_value = fake_stats
        lge = League.from_database(last_season_factor=0.5)
        self.assertEqual(lge.last_season_stats, fake_stats)

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_last_season_stats', return_value=[])
    @patch('league.read_schedule')
    def test_missing_competition_name_falls_back_to_unknown(self, mock_rs, mock_rlss, mock_tfr):
        """If Competition_Name is absent from match rows, name defaults to 'Unknown League'."""
        row = _full_row('2024-08-01')
        row.pop('Competition_Name', None)
        row['Country'] = 'England'
        mock_rs.return_value = [row]
        lge = League.from_database()
        self.assertEqual(lge.name, 'Unknown League')

    @patch('league.Team.teams_from_results', return_value={})
    @patch('league.read_last_season_stats', return_value=[])
    @patch('league.read_schedule')
    def test_country_defaults_to_england_when_missing(self, mock_rs, mock_rlss, mock_tfr):
        """If Country is absent from match rows, it defaults to 'England' for the stats query."""
        row = _full_row('2024-08-01')
        row['Competition_Name'] = 'Premier League'
        row.pop('Country', None)
        mock_rs.return_value = [row]
        lge = League.from_database(last_season_factor=0.5)
        # The country fallback is used — just verify the call completed without error
        mock_rlss.assert_called_once()
        call_kwargs = mock_rlss.call_args
        self.assertIn('England', str(call_kwargs))


if __name__ == '__main__':
    unittest.main()
