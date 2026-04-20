import unittest
from team import Team


class TestTeamInit(unittest.TestCase):
    def setUp(self):
        self.team = Team('Arsenal')

    def test_name_stored(self):
        """Team stores the name passed to the constructor."""
        self.assertEqual(self.team.name, 'Arsenal')

    def test_integer_defaults(self):
        """All integer counter attributes initialise to zero."""
        int_attrs = [
            'home_games_played', 'away_games_played',
            'home_points', 'away_points',
            'home_goals', 'away_goals',
            'home_goals_against', 'away_goals_against',
        ]
        for attr in int_attrs:
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.team, attr), 0)

    def test_float_defaults(self):
        """All float attributes initialise to 0.0."""
        float_attrs = [
            'home_xg', 'away_xg', 'home_xga', 'away_xga',
            'home_attack_strength_cs', 'away_attack_strength_cs',
            'home_defence_strength_cs', 'away_defence_strength_cs',
            'home_attack_strength_ls', 'away_attack_strength_ls',
            'home_defence_strength_ls', 'away_defence_strength_ls',
            'home_attack_strength', 'away_attack_strength',
            'home_defence_strength', 'away_defence_strength',
        ]
        for attr in float_attrs:
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.team, attr), 0.0)


class TestTeamsFromResults(unittest.TestCase):
    # league averages hand-calculated from the two fixture rows (xG_factor=0.6):
    # smooth_home = 0.4*(2+0) + 0.6*(1.5+0.8) = 2.18 / 2 games = 1.09
    # smooth_away = 0.4*(1+2) + 0.6*(1.2+1.7) = 2.94 / 2 games = 1.47
    LEAGUE_AVG_HOME = 1.09
    LEAGUE_AVG_AWAY = 1.47

    def _two_match_results(self):
        return [
            {'Date': '2024-08-01', 'Home': 'A', 'Away': 'B',
             'HomeGoals': 2, 'AwayGoals': 1,
             'Home_xG': 1.5, 'Away_xG': 1.2,
             'Home_pts': 3, 'Away_pts': 0},
            {'Date': '2024-08-02', 'Home': 'B', 'Away': 'A',
             'HomeGoals': 0, 'AwayGoals': 2,
             'Home_xG': 0.8, 'Away_xG': 1.7,
             'Home_pts': 0, 'Away_pts': 3},
        ]

    def test_normal_data_creates_teams(self):
        """Two-match fixture produces Team instances keyed by team name."""
        teams = Team.teams_from_results(
            self._two_match_results(),
            self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY,
        )
        self.assertIn('A', teams)
        self.assertIn('B', teams)
        self.assertIsInstance(teams['A'], Team)
        self.assertIsInstance(teams['B'], Team)

    def test_normal_data_stat_accumulation(self):
        """Team A accumulates correct goals, xG, goals_against, and points across both matches."""
        teams = Team.teams_from_results(
            self._two_match_results(),
            self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY,
        )
        a = teams['A']
        self.assertEqual(a.home_games_played, 1)
        self.assertEqual(a.away_games_played, 1)
        self.assertEqual(a.home_goals, 2)
        self.assertEqual(a.away_goals, 2)
        self.assertEqual(a.home_goals_against, 1)
        self.assertEqual(a.away_goals_against, 0)
        self.assertAlmostEqual(a.home_xg, 1.5)
        self.assertAlmostEqual(a.away_xg, 1.7)
        self.assertAlmostEqual(a.home_xga, 1.2)
        self.assertAlmostEqual(a.away_xga, 0.8)
        self.assertEqual(a.home_points, 3)
        self.assertEqual(a.away_points, 3)

    def test_empty_results_no_ls_returns_empty(self):
        """Empty results with no last_season_strengths produces an empty dict."""
        teams = Team.teams_from_results(
            [], self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY,
        )
        self.assertEqual(teams, {})

    def test_empty_results_with_ls_creates_teams(self):
        """Empty results paired with last_season_strengths creates teams from ls data only."""
        ls = [
            {'team': 'X', 'home_attack_strength': 1.1, 'home_defence_strength': 0.9,
             'away_attack_strength': 1.0, 'away_defence_strength': 1.0},
        ]
        teams = Team.teams_from_results(
            [], self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY,
            last_season_factor=0.5, last_season_strengths=ls,
        )
        self.assertIn('X', teams)
        self.assertIsInstance(teams['X'], Team)

    def test_missing_required_key_raises_value_error(self):
        """A result row missing a required key raises ValueError that names the missing key."""
        bad = [{'Date': '2024-08-01', 'Home': 'A', 'Away': 'B',
                'HomeGoals': 1, 'AwayGoals': 0,
                'Away_xG': 1.0, 'Home_pts': 3, 'Away_pts': 0}]  # Home_xG missing
        with self.assertRaises(ValueError) as ctx:
            Team.teams_from_results(bad, self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY)
        self.assertIn('Home_xG', str(ctx.exception))

    def test_invalid_type_goals_raises_value_error(self):
        """Non-numeric HomeGoals value raises ValueError."""
        bad = [{'Date': '2024-08-01', 'Home': 'A', 'Away': 'B',
                'HomeGoals': 'abc', 'AwayGoals': 0,
                'Home_xG': 1.0, 'Away_xG': 1.0,
                'Home_pts': 3, 'Away_pts': 0}]
        with self.assertRaises(ValueError):
            Team.teams_from_results(bad, self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY)

    def test_invalid_type_xg_raises_value_error(self):
        """None as Home_xG raises ValueError."""
        bad = [{'Date': '2024-08-01', 'Home': 'A', 'Away': 'B',
                'HomeGoals': 1, 'AwayGoals': 0,
                'Home_xG': None, 'Away_xG': 1.0,
                'Home_pts': 3, 'Away_pts': 0}]
        with self.assertRaises(ValueError):
            Team.teams_from_results(bad, self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY)

    def test_no_lsf_final_strength_equals_cs(self):
        """Without last_season_factor each final strength equals its cs counterpart."""
        teams = Team.teams_from_results(
            self._two_match_results(),
            self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY,
        )
        for team in teams.values():
            self.assertAlmostEqual(team.home_attack_strength, team.home_attack_strength_cs)
            self.assertAlmostEqual(team.away_attack_strength, team.away_attack_strength_cs)
            self.assertAlmostEqual(team.home_defence_strength, team.home_defence_strength_cs)
            self.assertAlmostEqual(team.away_defence_strength, team.away_defence_strength_cs)

    def test_with_lsf_final_strength_is_blend(self):
        """With last_season_factor, final strength = cs*(1-lsf) + ls*lsf."""
        lsf = 0.3
        ls = [
            {'team': 'A', 'home_attack_strength': 1.2, 'home_defence_strength': 0.9,
             'away_attack_strength': 1.1, 'away_defence_strength': 0.8},
            {'team': 'B', 'home_attack_strength': 0.8, 'home_defence_strength': 1.1,
             'away_attack_strength': 0.9, 'away_defence_strength': 1.2},
        ]
        teams = Team.teams_from_results(
            self._two_match_results(),
            self.LEAGUE_AVG_HOME, self.LEAGUE_AVG_AWAY,
            last_season_factor=lsf,
            last_season_strengths=ls,
        )
        a = teams['A']
        expected_home_att = a.home_attack_strength_cs * 0.7 + a.home_attack_strength_ls * 0.3
        self.assertAlmostEqual(a.home_attack_strength, expected_home_att, places=9)
        expected_away_def = a.away_defence_strength_cs * 0.7 + a.away_defence_strength_ls * 0.3
        self.assertAlmostEqual(a.away_defence_strength, expected_away_def, places=9)


class TestCalculateTeamStrengths(unittest.TestCase):
    # Fixture values chosen so expected cs strengths are exact fractions:
    # Team T: 1 home game (2 goals, 1.5 xg scored; 1 ga, 1.2 xga)
    #         1 away game (1 goal, 1.2 xg scored; 2 ga, 1.5 xga)
    # league_avg_home=1.5, league_avg_away=1.2, xG_factor=0.6 (inverse=0.4)
    #
    # cs_home_att = (0.4*2 + 0.6*1.5) / (1*1.5) = 1.7/1.5  = 17/15
    # cs_home_def = (0.4*1 + 0.6*1.2) / (1*1.2) = 1.12/1.2 = 14/15
    # cs_away_att = (0.4*1 + 0.6*1.2) / (1*1.2) = 1.12/1.2 = 14/15
    # cs_away_def = (0.4*2 + 0.6*1.5) / (1*1.5) = 1.7/1.5  = 17/15

    def _make_team(self, hgp=1, agp=1, hg=2, ag=1,
                   hga=1, aga=2, hxg=1.5, axg=1.2, hxga=1.2, axga=1.5):
        t = Team('T')
        t.home_games_played = hgp
        t.away_games_played = agp
        t.home_goals = hg
        t.away_goals = ag
        t.home_goals_against = hga
        t.away_goals_against = aga
        t.home_xg = hxg
        t.away_xg = axg
        t.home_xga = hxga
        t.away_xga = axga
        return t

    def _call(self, team, league_avg_home=1.5, league_avg_away=1.2,
              xG_factor=0.6, last_season_factor=None,
              last_season_strengths=None, init=True):
        Team.calculate_team_strengths(
            {'T': team}, league_avg_home, league_avg_away,
            xG_factor, last_season_factor, last_season_strengths, init,
        )

    def test_formula_no_lsf(self):
        """Without lsf, cs and final strengths match the smoothed-goals formula exactly."""
        t = self._make_team()
        self._call(t)
        self.assertAlmostEqual(t.home_attack_strength_cs, 17/15, places=9)
        self.assertAlmostEqual(t.home_defence_strength_cs, 14/15, places=9)
        self.assertAlmostEqual(t.away_attack_strength_cs, 14/15, places=9)
        self.assertAlmostEqual(t.away_defence_strength_cs, 17/15, places=9)
        # No blending: final == cs
        self.assertAlmostEqual(t.home_attack_strength, 17/15, places=9)
        self.assertAlmostEqual(t.away_defence_strength, 17/15, places=9)

    def test_formula_with_lsf_blend(self):
        """With lsf=0.3, each final strength = cs*0.7 + ls*0.3."""
        t = self._make_team()
        t.home_attack_strength_ls = 1.2
        t.home_defence_strength_ls = 0.9
        t.away_attack_strength_ls = 1.1
        t.away_defence_strength_ls = 0.95
        self._call(t, last_season_factor=0.3)
        self.assertAlmostEqual(t.home_attack_strength, (17/15)*0.7 + 1.2*0.3, places=9)
        self.assertAlmostEqual(t.home_defence_strength, (14/15)*0.7 + 0.9*0.3, places=9)
        self.assertAlmostEqual(t.away_attack_strength, (14/15)*0.7 + 1.1*0.3, places=9)
        self.assertAlmostEqual(t.away_defence_strength, (17/15)*0.7 + 0.95*0.3, places=9)

    def test_zero_home_games_no_lsf_defaults_to_one(self):
        """home_games_played=0 without lsf falls back to neutral cs strengths of 1.0."""
        t = self._make_team(hgp=0)
        self._call(t)
        self.assertEqual(t.home_attack_strength_cs, 1.0)
        self.assertEqual(t.home_defence_strength_cs, 1.0)

    def test_zero_home_games_with_lsf_uses_ls(self):
        """home_games_played=0 with lsf sets home cs strengths to the ls values."""
        t = self._make_team(hgp=0)
        t.home_attack_strength_ls = 1.3
        t.home_defence_strength_ls = 0.85
        self._call(t, last_season_factor=0.3)
        self.assertAlmostEqual(t.home_attack_strength_cs, 1.3, places=9)
        self.assertAlmostEqual(t.home_defence_strength_cs, 0.85, places=9)

    def test_zero_away_games_no_lsf_defaults_to_one(self):
        """away_games_played=0 without lsf falls back to neutral cs strengths of 1.0."""
        t = self._make_team(agp=0)
        self._call(t)
        self.assertEqual(t.away_attack_strength_cs, 1.0)
        self.assertEqual(t.away_defence_strength_cs, 1.0)

    def test_zero_away_games_with_lsf_uses_ls(self):
        """away_games_played=0 with lsf sets away cs strengths to the ls values."""
        t = self._make_team(agp=0)
        t.away_attack_strength_ls = 0.95
        t.away_defence_strength_ls = 1.05
        self._call(t, last_season_factor=0.3)
        self.assertAlmostEqual(t.away_attack_strength_cs, 0.95, places=9)
        self.assertAlmostEqual(t.away_defence_strength_cs, 1.05, places=9)

    def test_zero_league_avg_home_no_division_error(self):
        """league_avg_home=0.0 triggers the else branch and does not raise ZeroDivisionError."""
        t = self._make_team()
        self._call(t, league_avg_home=0.0)  # must not raise

    def test_zero_league_avg_away_no_division_error(self):
        """league_avg_away=0.0 triggers the else branch and does not raise ZeroDivisionError."""
        t = self._make_team()
        self._call(t, league_avg_away=0.0)  # must not raise

    def test_lsf_out_of_range_high_raises(self):
        """last_season_factor > 1.0 raises ValueError."""
        t = self._make_team()
        with self.assertRaises(ValueError):
            self._call(t, last_season_factor=1.5)

    def test_lsf_out_of_range_low_raises(self):
        """last_season_factor < 0.0 raises ValueError."""
        t = self._make_team()
        with self.assertRaises(ValueError):
            self._call(t, last_season_factor=-0.1)

    def test_cs_zero_with_lsf_uses_ls_directly(self):
        """When cs is 0.0 (zero goals and zero xG scored), final strength equals ls directly, not a blend."""
        # Team played 1 home game but scored 0 goals with 0 xG -> smoothed = 0 -> cs = 0.0
        t = self._make_team(hg=0, hxg=0.0)
        t.home_attack_strength_ls = 0.9
        self._call(t, last_season_factor=0.5)
        # If blended: 0.0*0.5 + 0.9*0.5 = 0.45 — but code uses ls directly when cs==0.0
        self.assertAlmostEqual(t.home_attack_strength, 0.9, places=9)


class TestUpdateTeams(unittest.TestCase):
    def _make_teams(self):
        """Two teams (H and V) with one prior home match to start with non-zero strengths."""
        prior = [
            {'Date': '2024-08-01', 'Home': 'H', 'Away': 'V',
             'HomeGoals': 1, 'AwayGoals': 1,
             'Home_xG': 1.2, 'Away_xG': 1.0,
             'Home_pts': 1, 'Away_pts': 1},
        ]
        return Team.teams_from_results(prior, 1.5, 1.2, xG_factor=0.6)

    def _update_row(self):
        return {
            'Date': '2024-08-08', 'Home': 'H', 'Away': 'V',
            'HomeGoals': 2, 'AwayGoals': 0,
            'Home_xG': 1.8, 'Away_xG': 0.5,
            'Home_pts': 3, 'Away_pts': 0,
        }

    def test_home_stat_increments(self):
        """Home team's games_played, goals, xg, and points each increment by the match values."""
        teams = self._make_teams()
        h = teams['H']
        before = (h.home_games_played, h.home_goals, h.home_xg, h.home_points)
        Team.update_teams(teams, [self._update_row()], 1.5, 1.2, xG_factor=0.6)
        self.assertEqual(h.home_games_played, before[0] + 1)
        self.assertEqual(h.home_goals, before[1] + 2)
        self.assertAlmostEqual(h.home_xg, before[2] + 1.8)
        self.assertEqual(h.home_points, before[3] + 3)

    def test_away_stat_increments(self):
        """Away team's games_played, goals, xg, and points each increment by the match values."""
        teams = self._make_teams()
        v = teams['V']
        before = (v.away_games_played, v.away_goals, v.away_xg, v.away_points)
        Team.update_teams(teams, [self._update_row()], 1.5, 1.2, xG_factor=0.6)
        self.assertEqual(v.away_games_played, before[0] + 1)
        self.assertEqual(v.away_goals, before[1] + 0)
        self.assertAlmostEqual(v.away_xg, before[2] + 0.5)
        self.assertEqual(v.away_points, before[3] + 0)

    def test_goals_against_cross_assigned(self):
        """home_goals_against receives the away score, and away_goals_against receives the home score."""
        teams = self._make_teams()
        row = self._update_row()
        h, v = teams['H'], teams['V']
        before_hga = h.home_goals_against
        before_vga = v.away_goals_against
        Team.update_teams(teams, [row], 1.5, 1.2, xG_factor=0.6)
        self.assertEqual(h.home_goals_against, before_hga + row['AwayGoals'])
        self.assertEqual(v.away_goals_against, before_vga + row['HomeGoals'])

    def test_strengths_recalculated_after_update(self):
        """Strengths change after update because calculate_team_strengths is called on fresh totals."""
        teams = self._make_teams()
        h = teams['H']
        strength_before = h.home_attack_strength
        Team.update_teams(teams, [self._update_row()], 1.5, 1.2, xG_factor=0.6)
        self.assertIsInstance(h.home_attack_strength, float)
        self.assertGreater(h.home_attack_strength, 0.0)
        self.assertNotAlmostEqual(h.home_attack_strength, strength_before, places=6)

    def test_multiple_rows_accumulate(self):
        """Two update rows are both processed; home games_played increments by 2."""
        teams = self._make_teams()
        h = teams['H']
        before_gp = h.home_games_played
        rows = [
            self._update_row(),
            {'Date': '2024-08-15', 'Home': 'H', 'Away': 'V',
             'HomeGoals': 1, 'AwayGoals': 1,
             'Home_xG': 1.0, 'Away_xG': 1.0,
             'Home_pts': 1, 'Away_pts': 1},
        ]
        Team.update_teams(teams, rows, 1.5, 1.2, xG_factor=0.6)
        self.assertEqual(h.home_games_played, before_gp + 2)


if __name__ == '__main__':
    unittest.main()
