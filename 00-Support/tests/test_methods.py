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
from lstnat.raster import (nested_means, weighted_native, station_support, local_grid,
                           sample_raster, validate_fine, require_scale_support, SCALE_SUPPORT)
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

    def test_station_support_geodesic_sizes(self):
        from pyproj import Geod, Transformer
        geod = Geod(ellps="WGS84")
        for lat in [23.55, 26.0, 28.32]:
            for side in [10, 210, 230, 1000, 1210, 1500]:
                crs, square = station_support(118, lat, side)
                self.assertEqual(square.bounds, (-side/2, -side/2, side/2, side/2))
                inverse = Transformer.from_crs(crs, 4326, always_xy=True)
                # Independent ellipsoidal distances across both axes and each side.
                for a, b in [((-side/2, 0), (side/2, 0)), ((0, -side/2), (0, side/2)),
                             ((-side/2, side/2), (side/2, side/2)),
                             ((side/2, -side/2), (side/2, side/2))]:
                    lon1, lat1 = inverse.transform(*a)
                    lon2, lat2 = inverse.transform(*b)
                    distance = geod.inv(lon1, lat1, lon2, lat2)[2]
                    self.assertAlmostEqual(distance, side, delta=0.00002)

    def test_ground_windows_on_mercator_storage(self):
        from pyproj import Transformer
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        import geopandas as gpd
        from shapely.geometry import Point
        for lat in [23.55, 28.32]:
            x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(118, lat)
            crs, _ = station_support(118, lat, 1210)
            to_local = Transformer.from_crs(3857, crs, always_xy=True)
            tr = from_origin(x-1285, y+1285, 10, 10)
            rr, cc = np.indices((257, 257))
            local_x, _ = to_local.transform(tr.c+(cc+.5)*10, tr.f-(rr+.5)*10)
            # A 210-ground-metre-wide cool strip; 230 ground m must reach outside it.
            array = np.where(abs(local_x) < 105, 10, 30).astype("float32")
            with MemoryFile() as mem:
                with mem.open(driver="GTiff", width=257, height=257, count=1, dtype="float32",
                              crs=3857, transform=tr, nodata=-9999) as dst:
                    dst.write(array, 1)
                with mem.open() as src:
                    stations = gpd.GeoDataFrame({"station_id": ["a"], "UrG": [1]},
                                                geometry=[Point(118, lat)], crs=4326)
                    sampled = sample_raster(src, stations)
                    self.assertAlmostEqual(sampled.L230.iloc[0], (210*10+20*30)/230, delta=.7)
                    self.assertGreater(sampled.L230.iloc[0], 11)
                    self.assertAlmostEqual(sampled.L1210.iloc[0], (210*10+1000*30)/1210, delta=.15)
                    require_scale_support(sampled)
                    self.assertEqual(sampled.scale_support.iloc[0], SCALE_SUPPORT)
                    self.assertEqual(sampled.source_crs.iloc[0], "EPSG:3857")
                    grid, transform, local = local_grid(src, 118, lat)
                    self.assertEqual(grid.shape, (121, 121))
                    self.assertEqual(transform*(60.5, 60.5), (0, 0))

    def test_native_local_support_area(self):
        from pyproj import Transformer
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        for lat in [23.55, 28.32]:
            crs, square = station_support(118, lat, 1000)
            to_storage = Transformer.from_crs(crs, 3857, always_xy=True)
            split, y = to_storage.transform(250, 0)
            tr = from_origin(split-2000, y+2000, 2000, 2000)
            with MemoryFile() as mem:
                with mem.open(driver="GTiff", width=2, height=2, count=1, dtype="float32",
                              crs=3857, transform=tr) as src:
                    src.write(np.array([[10, 20], [10, 20]], dtype="float32"), 1)
                    self.assertAlmostEqual(weighted_native(src, square, crs), 12.5, delta=.001)

    def test_storage_crs_and_scale_provenance(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        for crs in [3857, 32650, 4326]:
            with MemoryFile() as mem:
                with mem.open(driver="GTiff", width=2, height=2, count=1, dtype="float32",
                              crs=crs, transform=from_origin(0, 20, 10, 10)) as src:
                    if crs == 4326:
                        with self.assertRaises(ValueError):
                            validate_fine(src)
                    else:
                        validate_fine(src)
        with self.assertRaisesRegex(ValueError, "rerun sample"):
            require_scale_support(pd.DataFrame({"L210": [20]}))
        from lstnat.scaling import scale_frame
        with self.assertRaisesRegex(ValueError, "rerun sample"):
            scale_frame(pd.DataFrame({"L210": [20]}), "TEM10", {})

    def test_local_sampling_nodata_and_extent(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        from pyproj import Transformer
        import geopandas as gpd
        from shapely.geometry import Point
        x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(118, 26)
        stations = gpd.GeoDataFrame({"station_id": ["a"], "UrG": [1]},
                                    geometry=[Point(118, 26)], crs=4326)
        for fill in [0, -9999]:
            with MemoryFile() as mem:
                with mem.open(driver="GTiff", width=51, height=51, count=1, dtype="float32",
                              crs=3857, transform=from_origin(x-255, y+255, 10, 10), nodata=-9999) as dst:
                    dst.write(np.full((51, 51), fill, dtype="float32"), 1)
                with mem.open() as src:
                    result = sample_raster(src, stations)
                    self.assertTrue(np.isnan(result.L1210.iloc[0]))
                    if fill == 0:
                        self.assertEqual(result.L210.iloc[0], 0)
                    else:
                        self.assertTrue(np.isnan(result.L10.iloc[0]))

    def test_scale_metadata_reaches_curves_and_figures(self):
        from lstnat.scaling import summarize_curves
        from lstnat import plotting
        from lstnat.settings import ROOT
        values = 1-np.exp(-np.array(SCALES)/200)
        data = pd.DataFrame({"time": "T1", "scenario": "original", "zone": "GA", "metric": "lee",
                             "scale_m": SCALES, "value": values, "scale_support": SCALE_SUPPORT})
        curves, knees = summarize_curves(data, {})
        self.assertTrue(curves.scale_support.eq(SCALE_SUPPORT).all())
        self.assertTrue(knees.scale_support.eq(SCALE_SUPPORT).all())
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            cfg = {"output_dir": str(Path(tmp).relative_to(ROOT))}
            curves.drop(columns="scale_support").to_csv(output_path(cfg, "scale", "curves.csv"), index=False)
            knees.to_csv(output_path(cfg, "scale", "knees.csv"), index=False)
            with self.assertRaisesRegex(ValueError, "rerun sample"):
                plotting.run(cfg)

    def test_manuscript_windows_and_scale_provenance(self):
        from pyproj import Transformer
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        import geopandas as gpd
        from shapely.geometry import Point
        from lstnat.raster import MANUSCRIPT_SUPPORT, configured_support
        self.assertEqual(configured_support({}), MANUSCRIPT_SUPPORT)
        for lat in [23.55, 28.32]:
            x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(118, lat)
            crs, square = station_support(118, lat, 1000, MANUSCRIPT_SUPPORT)
            self.assertEqual(crs.to_epsg(), 3857)
            np.testing.assert_allclose(square.bounds, [x-500, y-500, x+500, y+500])
            # A 210-coordinate-metre cool strip centered on the station.
            grid = np.full((151, 151), 30, dtype='float32')
            grid[:, 65:86] = 10
            with MemoryFile() as mem:
                with mem.open(driver='GTiff', width=151, height=151, count=1, dtype='float32',
                              crs=3857, transform=from_origin(x-755, y+755, 10, 10)) as dst:
                    dst.write(grid, 1)
                with mem.open() as src:
                    stations = gpd.GeoDataFrame({'station_id': ['a'], 'UrG': [1]},
                                               geometry=[Point(118, lat)], crs=4326)
                    sampled = sample_raster(src, stations, MANUSCRIPT_SUPPORT)
                    self.assertAlmostEqual(sampled.L210.iloc[0], 10)
                    self.assertAlmostEqual(sampled.L230.iloc[0], (210*10+20*30)/230)
                    self.assertEqual(sampled.resampling.iloc[0], 'nearest')
                    require_scale_support(sampled, MANUSCRIPT_SUPPORT)
                    with self.assertRaisesRegex(ValueError, 'incompatible'):
                        require_scale_support(sampled, SCALE_SUPPORT)
        mixed = pd.DataFrame({'scale_support': [MANUSCRIPT_SUPPORT, SCALE_SUPPORT]})
        with self.assertRaises(ValueError):
            require_scale_support(mixed)

    def test_coarse_export_grid_validation(self):
        import rasterio
        from rasterio.transform import from_origin
        from lstnat.downscaling import prepare_coarse
        from lstnat.settings import ROOT
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            native, emb = Path(tmp)/'native.tif', Path(tmp)/'emb.tif'
            profile = dict(driver='GTiff', width=2, height=2, dtype='float32', crs=3857,
                           transform=from_origin(0, 2000, 1000, 1000))
            with rasterio.open(native, 'w', count=1, **profile) as dst:
                dst.write(np.ones((2, 2), dtype='float32'), 1)
            with rasterio.open(emb, 'w', count=64, **profile) as dst:
                dst.write(np.ones((64, 2, 2), dtype='float32'))
            cfg = {'inputs': {'native_lst': {'T1': str(native)}, 'embedding_coarse': {'T1': str(emb)}}}
            self.assertEqual(prepare_coarse(cfg, 'T1'), emb)
            # A shifted coarse grid must not silently become a training input.
            profile['transform'] = from_origin(10, 2000, 1000, 1000)
            with rasterio.open(emb, 'w', count=64, **profile) as dst:
                dst.write(np.ones((64, 2, 2), dtype='float32'))
            with self.assertRaisesRegex(ValueError, 'exact native TRIMS grid'):
                prepare_coarse(cfg, 'T1')

    def test_zone_figures_use_equal_size_estimates(self):
        from unittest.mock import patch
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.axes import Axes
        from lstnat.settings import ROOT, TIMES, ZONES
        from lstnat.plotting import run
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            cfg = {'output_dir': str(Path(tmp).relative_to(ROOT))}
            raw = pd.DataFrame([{'time': t, 'zone': z, 'rho': .2, 'partial_rho': .1}
                                for z in ['GA', *ZONES.values()] for t in TIMES])
            balanced = pd.DataFrame([{'time': t, 'zone': z, 'balanced_rho': .8,
                                     'balanced_partial_rho': .6}
                                    for z in ZONES.values() for t in TIMES])
            raw.to_csv(output_path(cfg, 'association', 'correlations.csv'), index=False)
            balanced.to_csv(output_path(cfg, 'association', 'balanced_correlations.csv'), index=False)
            plotted = []
            original = Axes.plot
            def capture(ax, *args, **kwargs):
                plotted.append(np.asarray(args[1], dtype=float))
                return original(ax, *args, **kwargs)
            with patch.object(Axes, 'plot', capture):
                run(cfg)
            np.testing.assert_allclose(plotted[0], .2)
            np.testing.assert_allclose(plotted[1], .1)
            for i in range(2, 10, 2):
                np.testing.assert_allclose(plotted[i], .8)
                np.testing.assert_allclose(plotted[i+1], .6)

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
