import unittest

from map_matching import Fix
from route_network import build_network


class RouteNetworkTests(unittest.TestCase):
    def test_repeated_passes_share_edges_and_owners(self):
        tracks = {
            "a": [Fix(37 + i * 0.0002, 55, i * 15) for i in range(8)],
            "b": [Fix(37 + i * 0.0002, 55.00004, i * 15) for i in range(8)],
        }
        network = build_network(tracks)
        self.assertTrue(network["edges"])
        self.assertTrue(any(set(owners) == {"a", "b"} for _, _, owners in network["edges"]))

    def test_time_gap_does_not_create_a_connector(self):
        network = build_network({"a": [Fix(37, 55, 0), Fix(37.0002, 55, 15),
                                       Fix(37.01, 55, 500), Fix(37.0102, 55, 515)]})
        for a, b, _ in network["edges"]:
            self.assertLess(abs(network["nodes"][a][1] - network["nodes"][b][1]), 0.001)


if __name__ == "__main__":
    unittest.main()
