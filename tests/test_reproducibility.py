import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import world_cup_simulation as sim
ROOT=Path(__file__).resolve().parents[1]

class HistoricalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ratings,cls.elo=sim.load_ratings(ROOT/'data'/'optimized_elo_2026.csv')

    def test_48_unique_teams(self):
        self.assertEqual(len(self.elo),48)
        self.assertEqual(len(set(sum(sim.GROUPS.values(),[]))),48)

    def test_match_probabilities(self):
        for a,b in [('Spain','Argentina'),('France','Haiti')]:
            v=sim.match_probs(a,b,self.elo)
            self.assertAlmostEqual(sum(v),1)
            self.assertTrue(all(0<=x<=1 for x in v))
            self.assertAlmostEqual(sum(sim.match_probs(b,a,self.elo)),1)

    def test_bracket_invariants(self):
        standings,qualified,thirds,r32,bracket,champ=sim.simulate_tournament(self.elo,seed=14)
        self.assertEqual(len(standings),48)
        self.assertEqual(len(qualified),32)
        self.assertEqual(len(thirds),12)
        self.assertEqual(len(r32),16)
        self.assertEqual(len(bracket),31)
        self.assertEqual(len(set(r32['Team A'])|set(r32['Team B'])),32)
        self.assertIn(champ,self.elo)

    def test_seeded_regression_against_archived_600(self):
        _,champ=sim.run_monte_carlo(self.elo,n_sims=600,seed=2026)
        old=pd.read_csv(ROOT/'data'/'legacy_monte_carlo_champion_probabilities.csv').set_index('Team')
        new=champ.set_index('Team')
        np.testing.assert_allclose(old.loc[new.index].select_dtypes('number'),new.select_dtypes('number'),rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(new['Champion %'].sum(),1)

if __name__=='__main__': unittest.main()
