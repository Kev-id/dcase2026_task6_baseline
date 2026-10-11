"""Guard against ambiguous matching and oracle/rounding interpretation errors."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('diagnostics',Path(__file__).resolve().parents[1]/'analysis/phase2_cpu_diagnostics.py')
d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class GeometryTests(unittest.TestCase):
    def test_zero_overlap_tie_uses_nearest_center(self):
        g=[[90,92],[20,22]]
        self.assertEqual(d.choose([10,12],g),[20,22])
        self.assertEqual(d.choose([10,12],g[::-1]),[20,22])

    def test_true_width_cannot_fix_distant_location(self):
        p=d.transform(100,2,True)
        self.assertEqual(d.iou(p,[30,32]),0)

    def test_both_oracles_recover_integer_target(self):
        g=[30,32]
        p=d.transform(d.center(g),d.width(g),True)
        self.assertEqual(d.iou(p,g),1)

    def test_shrinking_can_destroy_overlap(self):
        g=[0,2]
        original=[0,10]
        smaller=d.transform(d.center(original),2,False)
        self.assertGreater(d.iou(original,g),d.iou(smaller,g))

    def test_rounding_clipping_matches_documented_order(self):
        self.assertEqual(d.transform(.5,2,True),[0.,2.])
        self.assertEqual(d.transform(299.5,2,True),[298.,300.])


if __name__=='__main__':
    unittest.main()
