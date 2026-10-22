"""Small scientific invariants and regression checks, using synthetic data only."""
import unittest
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from lstnat.settings import SCALES, output_path
from lstnat.statistics import pair_filter, correlations, knee
from lstnat.spatial import adaptive_weights, lee
from lstnat.raster import nested_means, weighted_native
from lstnat.downscaling import block_offsets, ARCHITECTURES


class MethodsTests(unittest.TestCase):
    def test_scale_grid_and_windows(self):
        self.assertEqual((len(SCALES), SCALES[0], SCALES[-1]), (61, 10, 1210))
        grid = np.zeros((121, 121)); grid[60, 60] = 9
        values = nested_means(grid, 60, 60, SCALES)
        self.assertEqual(values["L10"], 9)
        self.assertEqual(values["L30"], 1)
        self.assertAlmostEqual(values["L1210"], 9/(121*121))
        self.assertTrue(np.isnan(nested_means(grid, 0, 0, SCALES)["L30"]))

    def test_pair_filter_order_independence(self):
        rng = np.random.default_rng(21)
        data = pd.DataFrame(rng.normal(size=(200, 2)), columns=["lst", "nat"])
        data.loc[0] = [100, 0]
        data.loc[1] = [0, np.inf]
        a = pair_filter(data, "lst", "nat")
        b = pair_filter(data, "nat", "lst")
        self.assertListEqual(a.index.tolist(), b.index.tolist())
        self.assertNotIn(0, a.index); self.assertNotIn(1, a.index)
        # Interval boundaries are retained, including a zero-IQR interval.
        data = pd.DataFrame({"lst": [1]*8+[9], "nat": [2]*9})
        self.assertEqual(len(pair_filter(data, "lst", "nat")), 8)

    def test_partial_rank_control(self):
        rng = np.random.default_rng(52)
        control = rng.normal(size=300)
        data = pd.DataFrame({"lst": control+rng.normal(0, .1, 300),
                             "nat": control+rng.normal(0, .1, 300), "control": control})
        result = correlations(data, "lst", "nat", ["control"])
        self.assertGreater(result["rho"], .9)
        self.assertLess(abs(result["partial_rho"]), .3)

    def test_knee_rule(self):
        x = np.array(SCALES)
        smooth, result = knee(x, 1-np.exp(-x/200))
        self.assertIsNotNone(result["scale"])
        self.assertTrue(10 < result["scale"] < 1210)
        _, flat = knee(x, np.ones(len(x)))
        self.assertIsNone(flat["scale"])
        _, linear = knee(x, x.astype(float))
        self.assertIsNone(linear["scale"])
        with self.assertRaises(ValueError):
            knee(x, np.full(len(x), np.nan))

    def test_weights_and_lee_decomposition(self):
        rng = np.random.default_rng(10)
        coords = rng.normal(size=(25, 2))
        w, meta = adaptive_weights(coords, 8)
        self.assertTrue(np.allclose(w.sum(axis=1), 1))
        self.assertTrue(np.allclose(w.diagonal(), 0))
        x = coords[:, 0] + rng.normal(0, .1, 25)
        y = x + rng.normal(0, .1, 25)
        value, p, local, lp = lee(x, y, w, permutations=19)
        self.assertAlmostEqual(value, local.mean(), places=10)
        self.assertGreaterEqual(p, 1/20)
        self.assertTrue((lp >= 1/20).all())
        # Verify arithmetic-mean symmetrization against an independent dense reference.
        from scipy.spatial.distance import cdist
        d = cdist(coords, coords)
        raw = np.zeros(d.shape)
        for i in range(len(coords)):
            ids = np.argsort(d[i])[1:9]
            raw[i, ids] = np.exp(-.5*(d[i, ids]/d[i, ids].max())**2)
        sym = (raw+raw.T)/2
        expected = sym/sym.sum(axis=1)[:, None]
        self.assertTrue(np.allclose(w.toarray(), expected))

    def test_residual_closure(self):
        offsets = block_offsets([20, 30, np.nan], np.array([30., 40., 0.]), np.array([2, 2, 0]))
        corrected = np.array([10, 20, 15, 25])+offsets[[0, 0, 1, 1]]
        self.assertAlmostEqual(corrected[:2].mean(), 20)
        self.assertAlmostEqual(corrected[2:].mean(), 30)
        self.assertTrue(np.isnan(offsets[2]))
        self.assertEqual(ARCHITECTURES["T4"]["units"], [384, 768, 384, 128])

    def test_native_area_weighting_and_nodata(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        from shapely.geometry import box
        array = np.array([[10., 20.], [0., -9999]], dtype="float32")
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=2, height=2, count=1, dtype="float32", crs=3857,
                          transform=from_origin(0, 2, 1, 1), nodata=-9999) as src:
                src.write(array, 1)
                self.assertAlmostEqual(weighted_native(src, box(.5, 1, 2, 2)), (10*.5+20)/1.5)
                self.assertTrue(np.isnan(weighted_native(src, box(0, 0, 2, 1))))

    def test_output_containment(self):
        with self.assertRaises(ValueError):
            output_path({"output_dir": "../Relationship"}, "test.csv")
        with self.assertRaises(ValueError):
            output_path({"output_dir": "results"}, "../../escape.csv")

    def test_coarse_pixel_assignment(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        from rasterio.windows import Window
        from lstnat.downscaling import coarse_ids
        with MemoryFile() as fine_mem, MemoryFile() as coarse_mem:
            with fine_mem.open(driver="GTiff", width=4, height=4, count=1, dtype="float32", crs=3857,
                               transform=from_origin(0, 40, 10, 10)) as fine, coarse_mem.open(
                               driver="GTiff", width=2, height=2, count=1, dtype="float32", crs=3857,
                               transform=from_origin(0, 40, 20, 20)) as coarse:
                actual = coarse_ids(fine, Window(0, 0, 4, 4), coarse)
                expected = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
                np.testing.assert_array_equal(actual, expected)

    def test_pixel_noise_and_overlap(self):
        import rasterio
        from rasterio.transform import from_origin
        from lstnat.sensitivity import perturb_raster
        from lstnat.raster import sample_raster
        import geopandas as gpd
        from shapely.geometry import Point
        from rasterio.io import MemoryFile
        with MemoryFile() as source_mem, MemoryFile() as noise_mem:
            source, perturbed = source_mem.name, noise_mem.name
            profile = dict(driver="GTiff", width=151, height=151, count=1, dtype="float32", crs=3857,
                           transform=from_origin(0, 1510, 10, 10), nodata=-9999)
            with rasterio.open(source, "w", **profile) as dst:
                dst.write(np.full((151, 151), 20, dtype="float32"), 1)
                dst.update_tags(residual_corrected="true")
            perturb_raster(source, perturbed, .1, 12)
            with rasterio.open(perturbed) as src:
                noise = src.read(1)/20-1
                self.assertLess(abs(float(noise.mean())), .003)
                self.assertAlmostEqual(float(noise.std()), .1, delta=.003)
                # Duplicate locations here test overlap consistency, not station-frame validity.
                stations = gpd.GeoDataFrame({"station_id": ["a", "b"], "UrG": [1, 2]},
                                            geometry=[Point(755, 755), Point(755, 755)], crs=3857)
                sampled = sample_raster(src, stations)
                np.testing.assert_allclose(sampled.iloc[0][[f"L{s}" for s in SCALES]].astype(float),
                                           sampled.iloc[1][[f"L{s}" for s in SCALES]].astype(float))
                self.assertLess(abs(sampled.L1210.iloc[0]-20), abs(sampled.L10.iloc[0]-20))


if __name__ == "__main__":
    unittest.main()
