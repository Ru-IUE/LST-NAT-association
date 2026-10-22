"""Time-specific DNNs, blockwise inference, and coarse-cell residual correction."""
from contextlib import ExitStack
import json
import numpy as np
from .settings import TIMES, input_path, output_path, setting, save_run

ARCHITECTURES = {
    "T1": dict(units=[256, 512, 256, 128], dropout=[.15]*4, batch=768, epochs=300, lr=.001, patience=30, lr_patience=15),
    "T2": dict(units=[256, 512, 256, 128], dropout=[.15]*4, batch=512, epochs=300, lr=.001, patience=30, lr_patience=15),
    "T3": dict(units=[256, 512, 256, 128], dropout=[.15]*4, batch=768, epochs=300, lr=.001, patience=30, lr_patience=15),
    "T4": dict(units=[384, 768, 384, 128], dropout=[.15, .20, .15, .15], batch=512, epochs=500, lr=.002, patience=50, lr_patience=25),
}


def open_bands(stack, path):
    """Accept one 64-band GeoTIFF or a directory with A1.tif ... A64.tif."""
    import rasterio
    if path.is_dir():
        sources = [(stack.enter_context(rasterio.open(path/f"A{i}.tif")), 1) for i in range(1, 65)]
    else:
        src = stack.enter_context(rasterio.open(path))
        if src.count != 64:
            raise ValueError("Expected exactly 64 embedding bands in A1...A64 order")
        sources = [(src, i) for i in range(1, 65)]
    first = sources[0][0]
    for src, _ in sources:
        if (src.crs, src.transform, src.shape) != (first.crs, first.transform, first.shape):
            raise ValueError("Embedding bands must share CRS, transform, and dimensions")
    return sources


def prepare_coarse(cfg, time):
    import rasterio
    from rasterio.warp import reproject, Resampling
    from .raster import validate_fine
    path = output_path(cfg, "preprocessing", f"{time}_embedding_mean.tif")
    with ExitStack() as stack:
        native = stack.enter_context(rasterio.open(input_path(cfg, "native_lst", time)))
        sources = open_bands(stack, input_path(cfg, "embedding_fine"))
        validate_fine(sources[0][0])
        profile = native.profile.copy()
        profile.update(driver="GTiff", count=64, dtype="float32", nodata=-9999, compress="deflate")
        with rasterio.open(path, "w", **profile) as dst:
            for i, (src, band) in enumerate(sources, 1):
                arr = np.full(native.shape, -9999, dtype=np.float32)
                reproject(rasterio.band(src, band), arr, src_transform=src.transform, src_crs=src.crs,
                          src_nodata=src.nodata if src.nodata is not None else -9999,
                          dst_transform=native.transform, dst_crs=native.crs, dst_nodata=-9999,
                          resampling=Resampling.average)
                dst.write(arr, i)
            dst.update_tags(aggregation="spatial_mean_from_10m", grid="native_TRIMS")
    return path


def metrics(observed, predicted):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    a, b = np.asarray(observed).ravel(), np.asarray(predicted).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        raise ValueError("Insufficient nonconstant matched samples for accuracy assessment")
    return {"n": len(a), "r2": float(r2_score(a, b)), "mae": float(mean_absolute_error(a, b)),
            "rmse": float(np.sqrt(mean_squared_error(a, b))),
            "pearson_r": float(pearsonr(a, b).statistic),
            "spearman_rho": float(spearmanr(a, b).statistic),
            "slope": float(np.polyfit(a, b, 1)[0])}


def train(cfg, times):
    import rasterio
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    for time in times:
        tf.keras.utils.set_random_seed(3407)
        if cfg.get("training", {}).get("deterministic", True):
            tf.config.experimental.enable_op_determinism()
        coarse_path = prepare_coarse(cfg, time)
        with rasterio.open(input_path(cfg, "native_lst", time)) as native, rasterio.open(coarse_path) as emb:
            y = native.read(1, masked=True).astype(float).filled(np.nan)
            x = emb.read(masked=True).astype(float).filled(np.nan).transpose(1, 2, 0)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=2) & (x != -9999).all(axis=2)
        x, y = x[valid].astype(np.float32), y[valid].astype(np.float32)
        ids = np.flatnonzero(valid.ravel())
        tr, ev = train_test_split(np.arange(len(y)), test_size=.2, random_state=3407)
        spec = ARCHITECTURES[time]
        layers = [keras.layers.Input(shape=(64,))]
        for units, dropout in zip(spec["units"], spec["dropout"]):
            layers.extend([keras.layers.Dense(units, activation="relu"), keras.layers.BatchNormalization(),
                           keras.layers.Dropout(dropout)])
        layers.append(keras.layers.Dense(1))
        model = keras.Sequential(layers)
        optimizer = keras.optimizers.AdamW(learning_rate=spec["lr"], **({"weight_decay": 3e-4} if time == "T4" else {}))
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=spec["patience"], restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=spec["lr_patience"], min_lr=1e-6)]
        history = model.fit(x[tr], y[tr], validation_data=(x[ev], y[ev]), epochs=spec["epochs"],
                            batch_size=spec["batch"], callbacks=callbacks, verbose=2)
        model.save(output_path(cfg, "models", time+".keras"))
        predicted = model.predict(x[ev], batch_size=1024, verbose=0).ravel()
        import pandas as pd
        pd.DataFrame({"native_flat_index": ids[ev], "observed": y[ev], "predicted": predicted}).to_csv(
            output_path(cfg, "validation", time+"_evaluation.csv"), index=False)
        pd.DataFrame(history.history).to_csv(output_path(cfg, "validation", time+"_history.csv"), index=False)
        score = {**metrics(y[ev], predicted), "training_n": len(tr), "evaluation_used_for_early_stopping": True,
                 "independent_10m_validation": False, "architecture": spec,
                 "optimizer": keras.optimizers.serialize(optimizer)}
        output_path(cfg, "validation", time+"_model_metrics.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
        save_run(cfg, "train", time=time)


def coarse_ids(fine, window, coarse):
    """Coarse pixel assignment by fine-pixel centre, consistently used in both passes."""
    from rasterio.warp import transform
    tr = fine.window_transform(window)
    r, c = np.indices((int(window.height), int(window.width)), dtype=float)
    x = tr.c + tr.a*(c+.5) + tr.b*(r+.5)
    y = tr.f + tr.d*(c+.5) + tr.e*(r+.5)
    xx, yy = transform(fine.crs, coarse.crs, x.ravel(), y.ravel())
    inv = ~coarse.transform
    xx, yy = np.asarray(xx), np.asarray(yy)
    cc = np.floor(inv.a*xx + inv.b*yy + inv.c).astype(np.int64)
    rr = np.floor(inv.d*xx + inv.e*yy + inv.f).astype(np.int64)
    valid = (rr >= 0) & (rr < coarse.height) & (cc >= 0) & (cc < coarse.width)
    ids = rr * coarse.width + cc
    ids[~valid] = -1
    return ids.reshape(r.shape)


def block_offsets(observed, sums, counts):
    observed = np.asarray(observed, float).ravel()
    result = np.full(observed.size, np.nan)
    valid = np.isfinite(observed) & (counts > 0)
    result[valid] = observed[valid] - sums[valid] / counts[valid]
    return result


def predict(cfg, times):
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform_bounds
    from tensorflow import keras
    from .raster import validate_fine
    for time in times:
        model = keras.models.load_model(output_path(cfg, "models", time+".keras"), compile=False)
        raw_path = output_path(cfg, "downscaling", time+"_uncorrected.tif")
        corrected_path = output_path(cfg, "downscaling", time+"_corrected.tif")
        with ExitStack() as stack:
            sources = open_bands(stack, input_path(cfg, "embedding_fine"))
            fine = sources[0][0]
            validate_fine(fine)
            coarse = stack.enter_context(rasterio.open(input_path(cfg, "native_lst", time)))
            bounds = transform_bounds(coarse.crs, fine.crs, *coarse.bounds, densify_pts=41)
            if fine.bounds.left > bounds[0]+10 or fine.bounds.bottom > bounds[1]+10 or fine.bounds.right < bounds[2]-10 or fine.bounds.top < bounds[3]-10:
                raise ValueError("Fine imagery must cover the entire native raster extent for full-block correction")
            observed = coarse.read(1, masked=True).astype(float).filled(np.nan)
            count = observed.size
            sums, counts = np.zeros(count), np.zeros(count, dtype=np.int64)
            profile = fine.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="float32", nodata=-9999,
                           tiled=True, blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER")
            with rasterio.open(raw_path, "w", **profile) as dst:
                for r in range(0, fine.height, 256):
                    for c in range(0, fine.width, 256):
                        win = Window(c, r, min(256, fine.width-c), min(256, fine.height-r))
                        cube = np.stack([src.read(b, window=win, masked=True).astype(float).filled(np.nan)
                                         for src, b in sources], axis=-1)
                        valid = np.isfinite(cube).all(axis=2) & (cube != -9999).all(axis=2)
                        values = np.full(valid.shape, -9999, dtype=np.float32)
                        if valid.any():
                            values[valid] = model.predict(cube[valid].astype(np.float32), batch_size=4096, verbose=0).ravel()
                        ids = coarse_ids(fine, win, coarse)
                        mask = valid & (ids >= 0) & np.isfinite(values)
                        np.add.at(sums, ids[mask], values[mask].astype(float))
                        np.add.at(counts, ids[mask], 1)
                        dst.write(values, 1, window=win)
                dst.update_tags(residual_corrected="false", time=time)
            offsets = block_offsets(observed, sums, counts)
            checksums = np.zeros(count)
            with rasterio.open(raw_path) as raw, rasterio.open(corrected_path, "w", **profile) as dst:
                for _, win in raw.block_windows(1):
                    values = raw.read(1, window=win, masked=True).astype(float).filled(np.nan)
                    ids = coarse_ids(raw, win, coarse)
                    valid = np.isfinite(values) & (ids >= 0)
                    delta = np.full(values.shape, np.nan)
                    delta[valid] = offsets[ids[valid]]
                    corrected = (values+delta).astype(np.float32)
                    good = np.isfinite(corrected)
                    np.add.at(checksums, ids[good], corrected[good].astype(float))
                    dst.write(np.where(good, corrected, -9999), 1, window=win)
                dst.update_tags(residual_corrected="true", block_assignment="fine_pixel_center", time=time)
            valid = np.isfinite(offsets)
            error = np.abs(checksums[valid]/counts[valid] - observed.ravel()[valid])
            maximum = float(error.max()) if len(error) else None
            if maximum is None or maximum > 1e-4:
                raise ArithmeticError(f"Residual correction closure failed: {maximum}")
            check = {"max_absolute_coarse_mean_error_c": maximum, "corrected_coarse_cells": int(valid.sum()),
                     "valid_native_without_fine_support": int(np.sum(np.isfinite(observed.ravel()) & (counts == 0))),
                     "assignment": "fine pixel centers; means over valid fine predictions", "float32_tolerance_c": 1e-4}
            output_path(cfg, "validation", time+"_correction_closure.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
        save_run(cfg, "predict", time=time)


def validate_product(cfg):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform
    from .raster import read_stations, validate_fine
    import pandas as pd
    stations = read_stations(input_path(cfg, "stations"))
    scores = []
    for time in TIMES:
        records = []
        with rasterio.open(output_path(cfg, "downscaling", time+"_uncorrected.tif")) as fine, rasterio.open(input_path(cfg, "native_lst", time)) as native:
            validate_fine(fine)
            for sid, p in zip(stations.station_id, stations.to_crs(fine.crs).geometry):
                win = from_bounds(p.x-750, p.y-750, p.x+750, p.y+750, fine.transform).round_offsets().round_lengths()
                arr = fine.read(1, window=win, masked=True).astype(float).filled(np.nan)
                rr, cc = np.indices(arr.shape, dtype=float)
                tr = fine.window_transform(win)
                xx, yy = tr.c+(cc.ravel()+.5)*tr.a, tr.f+(rr.ravel()+.5)*tr.e
                xx, yy = transform(fine.crs, native.crs, xx, yy)
                samples = np.ma.concatenate(list(native.sample(zip(xx, yy), masked=True))).astype(float).filled(np.nan).ravel()
                valid = np.isfinite(arr.ravel()) & np.isfinite(samples)
                if valid.any():
                    records.append({"station_id": sid, "observed": float(samples[valid].mean()),
                                    "predicted": float(arr.ravel()[valid].mean()), "valid_pixels": int(valid.sum())})
        table = pd.DataFrame(records)
        score = metrics(table.observed, table.predicted)
        scores.append({"time": time, **score})
        table.to_csv(output_path(cfg, "validation", time+"_station_windows.csv"), index=False)
    pd.DataFrame(scores).to_csv(output_path(cfg, "validation", "product_metrics.csv"), index=False)
    save_run(cfg, "validate-product", corrected=False, independent_10m_validation=False)
