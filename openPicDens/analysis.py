"""
Main analysis class for porosity ring profiling.

The public class :class:`PICDens` scans micrograph directories,
computes porosity profiles, and writes chronology files.

The binarisation strategy is injected via a :class:`~openPicDens.binarization.Binarizer`
instance, keeping image processing concerns separate from
dendrochronological analysis::

    from openPicDens.binarization import Binarizer
    from openPicDens.analysis import PICDens

    bi = Binarizer(method="Otsu", blur="Median")
    scan = PICDens(binarizer=bi, save_path="results/", ...)
    scan.startScan()
"""

from __future__ import annotations

import os
from itertools import groupby
from statistics import mean, median

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from .binarization import Binarizer
from .config import SAVE_NAMES, SAVE_PATHS, SAVE_RWL_NAMES
from .io import rw2rwl, save_df
from .utils import (
    getNormalisationPorosityProfile,
    getTreeDirs,
    initPath,
    initResultsPathsFromImages,
    mathRound,
    pad_dict_list,
    smaDF,
)


class PICDens:
    """Measure porosity ring profiles from wood micrograph images.

    Parameters
    ----------
    binarizer : Binarizer
        Configured binarisation strategy. Decouple image preprocessing
        from analysis by constructing the :class:`Binarizer` separately
        and passing it here.
    save_path : str
        Directory where all results will be written.
    root : str
        Directory containing per-tree sub-directories with images.
        When *use_pred_bi_imgs* is ``True`` this is ignored and
        *bi_imgs_path* is used instead.
    species_name : str
        Species identifier (used in metadata).
    year_start : int
        Most recent calendar year in the chronology.
    norm_number : int
        Target length for normalised porosity profiles.
    pix_to_mcm_coef : float | int
        Pixel-to-micrometre conversion coefficient (default: ``1``).
    norm_method : str
        Normalisation method: ``"small_ring"``, ``"median"``,
        ``"mean"``, or ``"normNumber"`` (default: ``"median"``).
    sma_interval : int
        Smoothing window for the Simple Moving Average (default: ``90``).
    gap_value : int | None
        Maximum white-pixel run kept by the gap filter.
        ``None`` disables it (default: ``None``).
    use_pred_bi_imgs : bool
        If ``True``, images in *bi_imgs_path* are used as-is without
        re-binarising (default: ``False``).
    bi_imgs_path : str | None
        Path to pre-binarised images; used when *use_pred_bi_imgs*
        is ``True`` (default: ``None``).
    """

    _N_SECTORS = 10

    def __init__(
        self,
        binarizer: Binarizer,
        save_path: str,
        root: str,
        species_name: str,
        year_start: int,
        norm_number: int,
        pix_to_mcm_coef: float | int = 1,
        norm_method: str = "median",
        sma_interval: int = 90,
        gap_value: int | None = None,
        use_pred_bi_imgs: bool = False,
        bi_imgs_path: str | None = None,
    ) -> None:
        self.binarizer = binarizer
        self.save_path = save_path
        self.root = bi_imgs_path if use_pred_bi_imgs else root
        self.species_name = species_name
        self.year_start = year_start
        self.norm_number = norm_number
        self.pix_to_mcm_coef = pix_to_mcm_coef
        self.norm_method = norm_method
        self.sma_interval = sma_interval
        self.gap_value = gap_value
        self.use_pred_bi_imgs = use_pred_bi_imgs
        self.bi_imgs_path = bi_imgs_path

        self._save_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def startScan(self) -> None:
        """Run a full scan of all tree directories and write results."""
        trees = getTreeDirs(self.root)
        paths = self._build_save_paths()
        initResultsPathsFromImages(root=self.save_path, trees_path=self.root)

        acc: dict[str, dict] = {
            k: {} for k in (
                "rw", "maxP", "minP", "meanP",
                "maxQP", "minQP", "meanQP",
                "ew", "lw", "ewpr", "lwpr", "ewPor", "lwPor",
            )
        }
        raw_acc: dict[str, pd.DataFrame] = {}
        sectors_acc: list[dict] = [{} for _ in range(self._N_SECTORS)]

        for tree_id, img_names in trees.items():
            tree_dir = os.path.join(self.root, tree_id)

            porosity_df = self.scanSubDir(tree_dir, img_names)
            porosity_sma = smaDF(porosity_df, self.sma_interval)
            raw_acc[tree_id] = porosity_sma

            acc["rw"][tree_id] = self.getRW(porosity_df)
            acc["maxP"][tree_id], acc["minP"][tree_id], acc["meanP"][tree_id] = \
                self.getPorosityCharacteristics(porosity_sma)
            acc["maxQP"][tree_id], acc["minQP"][tree_id], acc["meanQP"][tree_id] = \
                self.getPorosityCharacteristicsProcentile(porosity_sma)
            acc["ew"][tree_id], acc["lw"][tree_id], \
                acc["ewpr"][tree_id], acc["lwpr"][tree_id], \
                acc["ewPor"][tree_id], acc["lwPor"][tree_id] = \
                self.getEarlyLateWidth(porosity_sma)

            for s, sector_data in enumerate(
                self.getSectorPorosity(porosity_sma, self._N_SECTORS)
            ):
                sectors_acc[s][tree_id] = sector_data

            nat_p = self.getPorProfilesNaturalValues(porosity_sma)
            norm_p = self.getNormPorosityProfiles(porosity_sma)
            save_df(nat_p,  os.path.join(paths["nat"],  f"{tree_id}.txt"))
            save_df(norm_p, os.path.join(paths["norm"], f"{tree_id}.txt"))
            save_df(porosity_df, os.path.join(paths["raw"], f"{tree_id}.txt"))

        # Long profiles
        _, long_df, avg, zlong_df, zlong_dict, zavg = \
            self.getLongPorosityProfile(raw_acc, "normNumber")
        avg  = self.getNormPorosityProfiles(avg)
        zavg = self.getNormPorosityProfiles(zavg)

        # Pad and convert accumulators to DataFrames
        dfs = {k: pd.DataFrame(pad_dict_list(v)) for k, v in acc.items()}

        # Save summary text files
        save_map = {
            "rw":    paths["rw"],   "maxP":  paths["maxP"],
            "minP":  paths["minP"], "meanP": paths["meanP"],
            "maxQP": paths["maxQP"],"minQP": paths["minQP"],
            "meanQP":paths["meanQP"],"ew":   paths["ew"],
            "lw":    paths["lw"],   "ewpr":  paths["ewpr"],
            "lwpr":  paths["lwpr"], "ewPor": paths["ewp"],
            "lwPor": paths["lwp"],
        }
        for key, txt_path in save_map.items():
            save_df(dfs[key], txt_path)

        save_df(long_df, paths["long"])
        save_df(zlong_df, paths["zlong"])
        save_df(avg, paths["avg"])
        save_df(zavg, os.path.join(self.save_path, "zAVG.txt"))

        for s in range(self._N_SECTORS):
            save_df(pd.DataFrame(pad_dict_list(sectors_acc[s])), paths["sec"][s])

        zpath = os.path.join(self.save_path, "zprofiles")
        initPath(zpath)
        for year, zdf in zlong_dict.items():
            save_df(zdf, os.path.join(zpath, f"{year}.txt"))

        # RWL export
        rwl_map = {
            "rw": (paths["rw_rwl"], 1),    "maxP":  (paths["max_rwl"],  1000),
            "minP":  (paths["min_rwl"],  1000), "meanP": (paths["mean_rwl"], 1000),
            "maxQP": (paths["maxq_rwl"], 1000), "minQP": (paths["minq_rwl"], 1000),
            "meanQP":(paths["meanq_rwl"],1000), "ew":    (paths["ew_rwl"],   1000),
            "lw":    (paths["lw_rwl"],   1000), "ewpr":  (paths["ewpr_rwl"], 1000),
            "lwpr":  (paths["lwpr_rwl"], 1000), "ewPor": (paths["ewp_rwl"],  1000),
            "lwPor": (paths["lwp_rwl"],  1000),
        }
        for key, (rwl_path, coef) in rwl_map.items():
            rw2rwl(dfs[key], rwl_path, end_year=self.year_start, coef=coef)

    def scanSubDir(self, sub_dir: str, img_names: list[str]) -> pd.DataFrame:
        """Scan all images in *sub_dir* and return per-year porosity profiles."""
        porosity_by_year: dict[int, list] = {}
        porosity_sd: dict[int, list] = {}
        porosity_sma_sd: dict[int, list] = {}

        tree_id = os.path.basename(sub_dir.rstrip("/\\"))
        sd_dir = os.path.join(self.save_path, SAVE_PATHS["sd_porosity_path"], tree_id)
        initPath(sd_dir)

        for img_name in img_names:
            ring_n = int(img_name.split(".")[0])
            img_path = os.path.join(sub_dir, img_name)

            porosity_df = self.scanImg(img_path)
            # porosity_df["finalPorosityProfile"] = porosity_df.mean(axis=1)

            sd_list, sma_sd_list, porosity_df = self.getSD(
                porosity_df=porosity_df, n_tree=ring_n, save_path=sd_dir,
            )
            porosity_sd[ring_n] = sd_list
            porosity_sma_sd[ring_n] = sma_sd_list
            porosity_by_year[self.year_start - ring_n + 1] = (
                porosity_df["finalPorosityProfile"].tolist()
            )

        save_df(pd.DataFrame(pad_dict_list(porosity_sd)),
                os.path.join(sd_dir, f"{tree_id}_SD.txt"))
        save_df(pd.DataFrame(pad_dict_list(porosity_sma_sd)),
                os.path.join(sd_dir, f"{tree_id}_SMA_SD.txt"))
        return pd.DataFrame(pad_dict_list(porosity_by_year))

    def scanImg(
        self,
        img_path: str,
        window_size: int = 1000,
        step: int = 200,
        windows_number: int = 5,
    ) -> pd.DataFrame:
        """Scan *img_path* with a sliding window and return porosity profiles."""
        print(f"[INFO] Scanning {img_path}")
        bi_img = self._load_binary_img(img_path)

        profiles: dict[int, list] = {}
        if window_size == -1:
            profiles[0] = self.getPorosityProfileToPix(bi_img)
        else:
            for i in range(windows_number):
                fragment = bi_img[i * step: i * step + window_size]
                profiles[i] = self.getPorosityProfileToPix(fragment)

        df = pd.DataFrame(profiles)
        df["finalPorosityProfile"] = df.mean(axis=1)
        return df

    def getPorosityProfileToPix(self, bi_img: np.ndarray) -> list[float]:
        """Compute a pixel-resolution porosity profile from *bi_img*."""
        transposed = bi_img.transpose()
        profile: list[float] = []
        for row in transposed:
            if self.gap_value:
                row = self.gapFilter(row)
            profile.append(float(np.sum(row == 255)) / len(row))
        return profile

    def gapFilter(self, scan_line: np.ndarray) -> np.ndarray:
        """Remove contiguous white runs longer than :attr:`gap_value`."""
        groups = [list(g) for _, g in groupby(scan_line)]
        filtered = [
            run for run in groups
            if not (run[0] == 255 and len(run) > self.gap_value)
        ]
        return np.array([px for run in filtered for px in run])

    def getSD(
        self,
        porosity_df: pd.DataFrame,
        n_tree: int,
        save_path: str,
        r: int = 300,
        step: int = 5,
        left_border_perc: float = 0.1,
        right_border_perc: float = 0.3,
    ) -> tuple[list, list, pd.DataFrame]:
        """Compute standard deviations at multiple smoothing scales."""
        profile = porosity_df["finalPorosityProfile"]
        sd_list: list[float] = []
        detrend_sd_list: list[float] = []
        detrend_df = pd.DataFrame({0: []})
        detrend_sma_df = pd.DataFrame({0: []})

        for window in range(5, r + 1, step):
            if len(profile.dropna()) < window:
                break
            smoothed = profile.rolling(window).mean().shift(periods=-(window // 2))
            porosity_df[f"smooth_{window}"] = smoothed

            left  = int(len(smoothed.dropna()) * left_border_perc)
            right = int(len(smoothed.dropna()) * right_border_perc)
            sd_list.append(smoothed.iloc[left:right].std())
            detrend_df[f"detrend_{window}"] = profile - smoothed

        if "detrend_300" in detrend_df.columns:
            detrend_sd_list.append(detrend_df["detrend_150"].std())
            for i in range(0, 300, step):
                col = detrend_df["detrend_150"].rolling(i).mean().shift(-(i // 2))
                detrend_sma_df[i] = col
                detrend_sd_list.append(col.std())

        save_df(porosity_df,  os.path.join(save_path, f"sd_{n_tree}.txt"))
        save_df(detrend_df,   os.path.join(save_path, f"detrend_{n_tree}.txt"))
        return sd_list, detrend_sd_list, porosity_df

    def zscore_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score every column of *df*."""
        return pd.DataFrame({
            col: stats.zscore(df[col].tolist()) for col in df.columns
        })

    def getLongPorosityProfile(
        self, porosity_dict: dict, norm_method: str
    ) -> tuple:
        """Assemble the long-term porosity profile across all trees and years."""
        year_end = min(
            min(df.columns) for df in porosity_dict.values()
        )
        long_dict: dict[int, dict] = {
            y: {} for y in range(self.year_start, year_end - 1, -1)
        }
        for year in range(year_end, self.year_start + 1):
            for tree_id, df in porosity_dict.items():
                if year in df.columns:
                    values = [
                        v for v in df[year].tolist() if str(v) != "nan"
                    ][::-1]
                    long_dict[year][tree_id] = values

        long_profiles: list[pd.DataFrame] = []
        zlong_dict: dict[int, pd.DataFrame] = {}
        means_dict: dict[int, list] = {}
        zmeans_dict: dict[int, list] = {}

        for year, year_data in long_dict.items():
            sub_df = pd.DataFrame(pad_dict_list(year_data))
            if sub_df.empty:
                continue
            norm_df = self.getNormPorosityDF(sub_df, norm_method)
            z_df    = self.zscore_df(norm_df)
            norm_df["MEAN"] = norm_df.mean(axis=1)
            z_df["MEAN"]    = z_df.mean(axis=1)
            means_dict[year]  = norm_df["MEAN"].tolist()
            zmeans_dict[year] = z_df["MEAN"].tolist()
            long_profiles.append(norm_df)
            zlong_dict[year] = z_df

        avg      = pd.DataFrame(pad_dict_list(means_dict))
        zavg     = pd.DataFrame(pad_dict_list(zmeans_dict))
        long_df  = pd.concat(long_profiles)
        zlong_df = pd.concat(zlong_dict.values())
        return long_profiles, long_df, avg, zlong_df, zlong_dict, zavg

    def getNormPorosityDF(
        self, porosity_df: pd.DataFrame, norm_method: str = "median"
    ) -> pd.DataFrame:
        """Normalise each column of *porosity_df* to a common length."""
        clean: dict[str, list] = {}
        lengths: list[int] = []
        for col in porosity_df.columns:
            values = [v for v in porosity_df[col].tolist() if str(v) != "nan"]
            if values:
                clean[col] = values
                lengths.append(len(values))

        if not lengths:
            return porosity_df

        req_len = mathRound({
            "small_ring": min(lengths),
            "median":     median(lengths),
            "mean":       mean(lengths),
            "normNumber": self.norm_number,
        }.get(norm_method, lengths[0]))

        return pd.DataFrame({
            col: (
                [0] * req_len
                if len(values) <= 5
                else getNormalisationPorosityProfile(values, req_len)
            )
            for col, values in clean.items()
        })

    def getPorProfilesNaturalValues(self, porosity_profiles: pd.DataFrame) -> pd.DataFrame:
        """Convert pixel-based profiles to physical length (micrometres)."""
        result: dict = {}
        for col in porosity_profiles.columns:
            values = [v for v in porosity_profiles[col].tolist() if str(v) != "nan"]
            if len(values) < 5:
                result[col] = [0] * self.norm_number
            else:
                req_len = mathRound(len(values) * self.pix_to_mcm_coef)
                result[col] = getNormalisationPorosityProfile(values, req_len)
        return pd.DataFrame(pad_dict_list(result))

    def getNormPorosityProfiles(self, porosity_profiles: pd.DataFrame) -> pd.DataFrame:
        """Normalise all profiles to :attr:`norm_number` points."""
        result: dict = {}
        for col in porosity_profiles.columns:
            values = [v for v in porosity_profiles[col].tolist() if str(v) != "nan"]
            result[col] = (
                [0] * self.norm_number
                if len(values) < 5
                else getNormalisationPorosityProfile(values, self.norm_number)
            )
        return pd.DataFrame(result)

    def getRW(self, porosity_profiles: pd.DataFrame) -> list[int]:
        """Extract ring-width chronology from porosity profiles."""
        columns = porosity_profiles.columns
        year_range = range(min(map(int, columns)), self.year_start + 1)
        trw = [
            -1 if y not in columns
            else int(porosity_profiles[y].count() * self.pix_to_mcm_coef)
            for y in year_range
        ]
        return list(map(int, trw))[::-1]

    def getPorosityCharacteristics(
        self, porosity_profiles: pd.DataFrame
    ) -> tuple[list, list, list]:
        """Return per-year maximum, minimum, and mean porosity."""
        return self._extract_year_stats(porosity_profiles, ("max", "min", "mean"))

    def getPorosityCharacteristicsProcentile(
        self, porosity_profiles: pd.DataFrame
    ) -> tuple[list, list, list]:
        """Return per-year 95th, 5th, and 50th percentile porosity."""
        return self._extract_year_stats(porosity_profiles, ("q95", "q05", "q50"))

    def getEarlyLateWidth(
        self, porosity_profiles: pd.DataFrame
    ) -> tuple[list, list, list, list, list, list]:
        """Separate each ring into earlywood and latewood."""
        columns = porosity_profiles.columns
        year_range = range(min(map(int, columns)), self.year_start + 1)
        ew, lw, ewpr, lwpr, ewpor, lwpor = [], [], [], [], [], []

        for year in year_range:
            if year not in columns or porosity_profiles[year].dropna().empty:
                for lst in (ew, lw, ewpr, lwpr, ewpor, lwpor):
                    lst.append(-1)
                continue

            clean = porosity_profiles[year].dropna()
            por_th = np.percentile(clean, 75)
            th_idx = next(
                (i for i in range(len(clean) - 1, 0, -1) if clean.iloc[i] >= por_th),
                0,
            )
            n = len(clean)
            ew.append(th_idx)
            lw.append(n - th_idx)
            ewpr.append(th_idx / n)
            lwpr.append(1 - th_idx / n)
            ewpor.append(clean.iloc[:th_idx].mean())
            lwpor.append(clean.iloc[th_idx:].mean())

        return (
            ew[::-1], lw[::-1], ewpr[::-1],
            lwpr[::-1], ewpor[::-1], lwpor[::-1],
        )

    def getSectorPorosity(
        self, porosity_profiles: pd.DataFrame, sectors_number: int = 10
    ) -> list[list]:
        """Compute mean porosity for each of *sectors_number* equal sectors."""
        columns = porosity_profiles.columns
        year_range = range(min(map(int, columns)), self.year_start + 1)
        sectors: list[list] = [[] for _ in range(sectors_number)]

        for year in year_range:
            if year not in columns or porosity_profiles[year].dropna().empty:
                for s in range(sectors_number):
                    sectors[s].append(-1)
                continue
            clean = porosity_profiles[year].dropna()
            step = len(clean) // sectors_number
            for s in range(sectors_number):
                start = step * s
                end = len(clean) - step * (sectors_number - 1 - s)
                sectors[s].append(clean.iloc[start:end].mean())

        return [s[::-1] for s in sectors]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_binary_img(self, img_path: str) -> np.ndarray:
        """Load *img_path* as a binary grayscale image."""
        if self.use_pred_bi_imgs:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, bi = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return bi
        return self.binarizer.binarize(img_path)

    def _extract_year_stats(
        self, porosity_profiles: pd.DataFrame, stat_keys: tuple[str, ...]
    ) -> tuple[list, ...]:
        """Generic per-year statistics extractor."""
        columns = porosity_profiles.columns
        year_range = range(min(map(int, columns)), self.year_start + 1)
        stat_map = {k: [] for k in stat_keys}
        dispatch = {
            "max":  lambda s: s.max(),
            "min":  lambda s: s.min(),
            "mean": lambda s: s.mean(),
            "q95":  lambda s: s.quantile(0.95),
            "q05":  lambda s: s.quantile(0.05),
            "q50":  lambda s: s.quantile(0.5),
        }
        for year in year_range:
            missing = year not in columns or porosity_profiles[year].dropna().empty
            for key in stat_keys:
                stat_map[key].append(
                    -1 if missing else dispatch[key](porosity_profiles[year])
                )
        return tuple(stat_map[k][::-1] for k in stat_keys)

    def _build_save_paths(self) -> dict:
        """Build the full dictionary of result file paths."""
        sp = self.save_path
        rwl_dir = os.path.join(sp, "rwl")
        sec_dir = os.path.join(sp, SAVE_PATHS["sec_path"])

        def p(name: str) -> str:
            return os.path.join(sp, SAVE_NAMES[name])

        def r(name: str) -> str:
            return os.path.join(rwl_dir, SAVE_RWL_NAMES[name])

        return {
            "nat":  os.path.join(sp, SAVE_PATHS["natural_path"]),
            "norm": os.path.join(sp, SAVE_PATHS["norm_path"]),
            "raw":  os.path.join(sp, SAVE_PATHS["raw_path"]),
            "rw": p("rw"), "maxP": p("max"), "minP": p("min"), "meanP": p("mean"),
            "maxQP": p("maxQ"), "minQP": p("minQ"), "meanQP": p("meanQ"),
            "ew": p("ew"), "lw": p("lw"),
            "ewpr": p("ewpr"), "lwpr": p("lwpr"),
            "ewp": p("ewp"), "lwp": p("lwp"),
            "long": p("long"), "zlong": p("zlong"), "avg": p("avg"),
            "sec": [
                os.path.join(sec_dir, SAVE_NAMES[s])
                for s in range(self._N_SECTORS)
            ],
            "rw_rwl": r("rw"), "max_rwl": r("max"), "min_rwl": r("min"),
            "mean_rwl": r("mean"), "maxq_rwl": r("maxQ"), "minq_rwl": r("minQ"),
            "meanq_rwl": r("meanQ"), "ew_rwl": r("ew"), "lw_rwl": r("lw"),
            "ewpr_rwl": r("ewpr"), "lwpr_rwl": r("lwpr"),
            "ewp_rwl": r("ewp"), "lwp_rwl": r("lwp"),
        }

    def _save_config(self) -> None:
        """Persist the current configuration to ``config.txt``."""
        initPath(self.save_path)
        save_df(pd.Series({
            "savePath":   self.save_path,
            "root":       self.root,
            "speciesName": self.species_name,
            "start year": self.year_start,
            "gap value":  self.gap_value,
            "SMA window": self.sma_interval,
            "binarizer":  repr(self.binarizer),
        }), os.path.join(self.save_path, "config.txt"))