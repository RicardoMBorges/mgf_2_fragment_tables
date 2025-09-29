import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# mgf_2_fragTable.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# MGF SURVEY
# =========================
# Standard library
import os
import re
import itertools
from collections import OrderedDict as OD
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    Union,
)

# Mass spectrometry I/O
from pyteomics import mgf as mgf_reader

# =========================
# Safe MGF loading
#    - uses pyteomics.mgf
#    - standardizes batch names: basename without extension
# =========================
def load_mgf_spectra(directory_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Safely load all .mgf files from a directory using pyteomics.mgf without indexing (use_index=False),
    which avoids the 'empty index' warning on some files. The batch name is standardized as the file
    basename without extension.

    Parameters
    ----------
    directory_path : str
        Path to the folder containing .mgf files.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A mapping: {batch_name -> list of spectra}, where each spectrum is a dict with:
        - "params": header parameters
        - "m/z array": list[float]
        - "intensity array": list[float]
    """
    spectra: Dict[str, List[Dict[str, Any]]] = {}
    for fn in os.listdir(directory_path):
        if not fn.lower().endswith(".mgf"):
            continue
        batch = os.path.splitext(fn)[0]
        spectra[batch] = []
        path = os.path.join(directory_path, fn)

        # No indexing to avoid empty-index warnings
        for spec in mgf_reader.read(path, use_index=False, convert_arrays=True):
            mz = spec.get("m/z array", [])
            ii = spec.get("intensity array", [])
            params = spec.get("params", {})
            spectra[batch].append({
                "params": params,
                "m/z array": list(map(float, mz)) if len(mz) else [],
                "intensity array": list(map(float, ii)) if len(ii) else [],
            })
    return spectra

# ---- main function ---------------------------------------------------------
def spectra_to_dataframe(
    spectra_by_batch: dict[str, list[dict]],
    *,
    top_n: int = 6,
    min_rel_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Build a DataFrame from spectra with:
      - 'batch', 'scans', 'scan_number'
      - 'precursor_mass'
      - 'fragments' (top_n peaks with ≥ min_rel_pct, formatted 'mz:rel%')
      - 'n_fragments'
    """
    rows = []
    for batch, spectra in spectra_by_batch.items():
        for spec in spectra:
            params = spec.get("params", {}) or {}
            mzs = spec.get("m/z array") or []
            intens = spec.get("intensity array") or []

            scans, scan_number = extract_scans_fields(params)
            precursor = get_precursor_mz(params)
            frag_str = select_fragments(mzs, intens, top_n=top_n, min_rel_pct=min_rel_pct)

            rows.append({
                "batch": batch,
                "scans": scans,
                "scan_number": scan_number,
                "precursor_mass": precursor,
                "fragments": frag_str,
                "n_fragments": frag_str.count(";") + 1 if frag_str else 0,
            })
    return pd.DataFrame(rows)

def extract_scans_fields(params: dict) -> tuple[str | None, int | None]:
    """Return ('scans' as-is, scan_number as int if parseable)."""
    scans_val = _first_param(params, ["SCANS","scans","scan","SCAN","scan_number","SCAN_NUMBER","FEATURE_ID","feature_id"])
    scan_number = None
    if scans_val is not None:
        s = str(scans_val).strip()
        m = re.search(r"\d+", s)
        if m:
            try:
                scan_number = int(m.group(0))
            except Exception:
                scan_number = None
    return scans_val, scan_number

# ---- helpers ---------------------------------------------------------------
def _first_param(params, keys):
    for k in keys:
        if k in params:
            v = params[k]
            if isinstance(v, (list, tuple)):
                return v[0] if v else None
            return v
    return None

def get_precursor_mz(params: dict) -> float | None:
    """Extract precursor m/z from PEPMASS (or fallbacks)."""
    # Try PEPMASS first (pyteomics often keeps the original case)
    for key in list(params.keys()):
        if key.lower() == "pepmass":
            v = params[key]
            # v can be: tuple/list (mz, intensity) OR string "mz intensity" OR a single float
            if isinstance(v, (tuple, list)):
                try: return float(v[0])
                except Exception: pass
            elif isinstance(v, str):
                toks = v.replace(",", " ").split()
                if toks:
                    try: return float(toks[0])
                    except Exception: pass
            else:
                try: return float(v)
                except Exception: pass

    # Fallback field names seen in the wild
    for alt in ("precursor_mz", "precursorMz", "parentmass", "ParentMass", "ms2precursor", "MS2Precursor"):
        if alt in params:
            v = params[alt]
            if isinstance(v, (tuple, list)):
                try: return float(v[0])
                except Exception: pass
            elif isinstance(v, str):
                toks = v.replace(",", " ").split()
                if toks:
                    try: return float(toks[0])
                    except Exception: pass
            else:
                try: return float(v)
                except Exception: pass
    return None

def extract_scans_fields(params: dict) -> tuple[str | None, int | None]:
    """Return ('scans' as-is, scan_number as int if parseable)."""
    scans_val = _first_param(params, ["SCANS","scans","scan","SCAN","scan_number","SCAN_NUMBER","FEATURE_ID","feature_id"])
    scan_number = None
    if scans_val is not None:
        s = str(scans_val).strip()
        m = re.search(r"\d+", s)
        if m:
            try:
                scan_number = int(m.group(0))
            except Exception:
                scan_number = None
    return scans_val, scan_number

def select_fragments(
    mzs: list[float] | np.ndarray,
    intens: list[float] | np.ndarray,
    *,
    top_n: int = 6,
    min_rel_pct: float = 1.0,
    mz_decimals: int = 4,
    rel_decimals: int = 1,
) -> str:
    """Return semicolon-separated 'mz:rel%' for peaks ≥ min_rel_pct, keeping top_n by rel intensity."""
    if mzs is None or intens is None:
        return ""
    if len(mzs) == 0 or len(intens) == 0 or len(mzs) != len(intens):
        return ""

    arr = np.column_stack([np.asarray(mzs, float), np.asarray(intens, float)])
    if arr.size == 0:
        return ""

    base = arr[:, 1].max()
    if not np.isfinite(base) or base <= 0:
        return ""

    rel = (arr[:, 1] / base) * 100.0
    keep = rel >= float(min_rel_pct)
    if not np.any(keep):
        return ""

    kept = np.column_stack([arr[keep, 0], rel[keep]])
    # sort by rel desc, tie-break by m/z asc
    order = np.lexsort((kept[:, 0], -kept[:, 1]))
    kept = kept[order][:top_n]

    parts = [f"{round(mz, mz_decimals)}:{round(r, rel_decimals)}" for mz, r in kept]
    return ";".join(parts)