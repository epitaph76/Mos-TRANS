import unittest

from map_matching import Fix, clean_fixes, match_fix, match_trace, simplify_route


class MapMatchingTests(unittest.TestCase):
    def test_invalid_and_impossible_ndtp_fixes_are_removed(self):
        fixes = [
            Fix(37.60, 55.75, 0),
            Fix(0, 0, 10, valid=False),
            Fix(37.85, 55.95, 20),
            Fix(37.601, 55.75, 30),
        ]
        cleaned = clean_fixes(fixes)
        self.assertEqual(len(cleaned), 2)
        self.assertAlmostEqual(cleaned[-1].lon, 37.601)

    def test_heading_resolves_opposing_route_segments(self):
        route = [(37.60, 55.75), (37.61, 55.75), (37.60, 55.75)]
        eastbound = match_fix(Fix(37.605, 55.75, 0, 30, 90), route)
        westbound = match_fix(Fix(37.605, 55.75, 0, 30, 270), route)
        self.assertIsNotNone(eastbound)
        self.assertIsNotNone(westbound)
        self.assertLess(eastbound.progress_m, westbound.progress_m)

    def test_route_continuity_prefers_forward_progress(self):
        route = [(37.60, 55.75), (37.61, 55.75), (37.60, 55.75)]
        matched = match_fix(Fix(37.605, 55.75, 0, 0), route, previous_progress_m=900)
        self.assertGreater(matched.progress_m, 900)

    def test_simplification_preserves_turn(self):
        fixes = [Fix(37.60, 55.75, 0), Fix(37.605, 55.75, 10), Fix(37.61, 55.75, 20), Fix(37.61, 55.755, 30)]
        route = simplify_route(fixes)
        self.assertEqual(len(route), 3)

    def test_trace_uses_latest_valid_fix(self):
        route = [(37.60, 55.75), (37.61, 55.75)]
        fixes = [Fix(37.602, 55.75, 0), Fix(0, 0, 10, valid=False), Fix(37.608, 55.75, 20)]
        matched = match_trace(fixes, route)
        self.assertAlmostEqual(matched.lon, 37.608, places=3)


if __name__ == "__main__":
    unittest.main()
