# streamlit app for MGF → fragments table
# Save as: app.py
# Run: streamlit run app.py

# --- Robust imports for cloud -----------------------------------------------
from pathlib import Path
import importlib.util
import tempfile
import shutil
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st

APP_VERSION = "v0.3 — optional MGF metadata / InChIKey support"

st.set_page_config(page_title="MGF → Fragment Tables", layout="wide")

st.markdown(
    """
    Upload **.mgf** files to extract fragments from MS/MS spectra.  

    Developed by **Ricardo M Borges** and **LAABio-IPPN-UFRJ**  
    contact: ricardo_mborges@yahoo.com.br  

    🔗 Details: [GitHub repository](https://github.com/RicardoMBorges/mgf_2_fragment_tables)

    [Tutorial](https://github.com/RicardoMBorges/mgf_2_fragment_tables/blob/main/README.md)

    Check also: [DAFdiscovery](https://dafdiscovery.streamlit.app/)
    
    Check also: [TLC2Chrom](https://tlc2chrom.streamlit.app/)
    """
)

# PayPal donate button
st.markdown(
    """
<hr>
<center>
<p>To support the app development:</p>
<a href="https://www.paypal.com/donate/?business=2FYTFNDV4F2D4&no_recurring=0&item_name=Support+with+%245+→+Send+receipt+to+tlc2chrom.app@gmail.com+with+your+login+email+→+Access+within+24h!&currency_code=USD" target="_blank">
    <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_SM.gif" alt="Donate with PayPal button" border="0">
</a>
</center>
""",
    unsafe_allow_html=True,
)


def _import_local_module(mod_name: str, base: Path):
    f = base / f"{mod_name}.py"
    if f.exists():
        spec = importlib.util.spec_from_file_location(mod_name, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    return None


# Verify third-party deps first (clear message if missing)
try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from pyteomics import mgf
except ModuleNotFoundError as e:
    st.error(
        "A required package is missing: "
        f"**{e.name}**. On Streamlit Cloud, make sure it's listed in `requirements.txt` "
        "(see the app README)."
    )
    st.stop()

HERE = Path(__file__).resolve().parent

try:
    import mgf_2_fragTable as mgf2frag  # noqa: E402
except ModuleNotFoundError:
    mgf2frag = _import_local_module("mgf_2_fragTable", HERE)

if mgf2frag is None:
    st.error("`mgf_2_fragTable.py` not found. Place it beside `app.py` in the repo.")
    st.stop()

from PIL import Image

STATIC_DIR = Path(__file__).parent / "static"
LOGO_PATH = STATIC_DIR / "LAABio.png"

try:
    logo = Image.open(LOGO_PATH)
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/LAABio.png")


# -----------------------------
# Metadata helpers
# -----------------------------
OPTIONAL_METADATA_FIELDS: Dict[str, List[str]] = {
    "compound_name": [
        "name",
        "compound_name",
        "compoundname",
        "compound",
        "title",
        "synonym",
    ],
    "inchikey": [
        "inchikey",
        "inchi_key",
        "inchi-key",
        "inchi key",
        "inchikey2d",
        "inchikey_2d",
        "inchikey14",
        "inchikey_14",
        "ik",
    ],
    "smiles": [
        "smiles",
        "canonical_smiles",
        "canonicalsmiles",
        "isomeric_smiles",
        "structure_smiles",
    ],
    "inchi": [
        "inchi",
        "standardinchi",
        "standard_inchi",
    ],
    "molecular_formula": [
        "formula",
        "molecular_formula",
        "molecularformula",
        "mol_formula",
    ],
    "exact_mass": [
        "exactmass",
        "exact_mass",
        "monoisotopic_mass",
        "monoisotopicmass",
    ],
    "adduct": [
        "adduct",
        "precursortype",
        "precursor_type",
        "iontype",
        "ion_type",
    ],
    "charge": [
        "charge",
    ],
    "rt_seconds": [
        "rtinseconds",
        "rt_in_seconds",
        "retentiontime_seconds",
    ],
    "rt_minutes": [
        "rtinminutes",
        "rt_in_minutes",
        "rt",
        "retentiontime",
        "retention_time",
    ],
    "collision_energy": [
        "collisionenergy",
        "collision_energy",
        "ce",
        "energy",
    ],
    "ion_mode": [
        "ionmode",
        "ion_mode",
        "polarity",
    ],
    "instrument": [
        "instrument",
        "instrumenttype",
        "instrument_type",
    ],
}


def _norm_key(key: Any) -> str:
    """Normalize MGF metadata keys for flexible matching.

    MGF exporters are inconsistent: INCHIKEY, INCHI_KEY, InChI-Key and
    inchi key should all match the same field.
    """
    import re
    return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())


def _norm_batch(value: Any) -> str:
    """Normalize file/batch labels so file.mgf and file can be matched."""
    txt = str(value).strip().replace("\\", "/").split("/")[-1]
    if txt.lower().endswith(".mgf"):
        txt = txt[:-4]
    return txt.lower()


def _clean_value(value: Any) -> str:
    """Convert MGF parameter values to a compact text representation."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value if v is not None)
    return str(value).strip()


def _first_available_param(params: Dict[str, Any], aliases: List[str]) -> str:
    """Return the first available value among alternative MGF parameter names."""
    normalized = {_norm_key(k): v for k, v in params.items()}
    for alias in aliases:
        value = normalized.get(_norm_key(alias), "")
        value = _clean_value(value)
        if value:
            return value
    return ""


def _parse_pepmass(value: Any) -> Optional[float]:
    """Extract precursor m/z from pyteomics PEPMASS values."""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            return float(value[0])
        txt = str(value).replace(",", " ").split()[0]
        return float(txt)
    except Exception:
        return None


def _metadata_records_from_mgf_file(mgf_path: Path) -> List[Dict[str, Any]]:
    """Read optional metadata from all BEGIN IONS blocks in one MGF file."""
    records: List[Dict[str, Any]] = []
    with mgf.read(str(mgf_path)) as reader:
        for i, spectrum in enumerate(reader, start=1):
            params = spectrum.get("params", {}) or {}
            record: Dict[str, Any] = {
                "batch": mgf_path.name,
                "batch_key": _norm_batch(mgf_path.name),
                "metadata_order": i,
                "metadata_scan_key": _clean_value(
                    params.get("scans")
                    or params.get("scan")
                    or params.get("scan_number")
                    or params.get("spectrumid")
                    or params.get("spectrum_id")
                    or i
                ),
                "metadata_precursor_mass": _parse_pepmass(
                    params.get("pepmass") or params.get("precursor_mass")
                ),
            }
            for out_col, aliases in OPTIONAL_METADATA_FIELDS.items():
                record[out_col] = _first_available_param(params, aliases)
            records.append(record)
    return records


@st.cache_data(show_spinner=False)
def extract_optional_metadata_from_dir(folder_with_mgfs: str) -> pd.DataFrame:
    """
    Extract optional annotations from MGF metadata.

    This complements mgf_2_fragTable.py without requiring changes to that module.
    If fields such as INCHIKEY, SMILES, FORMULA, RT or CE are absent, the
    corresponding cells remain blank.
    """
    folder = Path(folder_with_mgfs)
    records: List[Dict[str, Any]] = []
    for mgf_path in sorted(folder.glob("*.mgf")):
        try:
            records.extend(_metadata_records_from_mgf_file(mgf_path))
        except Exception as e:
            records.append(
                {
                    "batch": mgf_path.name,
                    "batch_key": _norm_batch(mgf_path.name),
                    "metadata_order": None,
                    "metadata_scan_key": "",
                    "metadata_precursor_mass": None,
                    "metadata_read_error": str(e),
                }
            )
    return pd.DataFrame(records)


def _add_merge_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Create robust keys for matching fragment table rows to MGF metadata."""
    out = df.copy()
    if "batch" not in out.columns:
        out["batch"] = ""
    out["batch_key"] = out["batch"].apply(_norm_batch)
    scan_source = None
    for col in ["scans", "scan_number", "scan", "spectrum_id"]:
        if col in out.columns:
            scan_source = col
            break
    out["metadata_scan_key"] = (
        out[scan_source].astype(str).str.strip() if scan_source else (out.index + 1).astype(str)
    )
    if "precursor_mass" in out.columns:
        out["metadata_precursor_mass"] = pd.to_numeric(out["precursor_mass"], errors="coerce")
    else:
        out["metadata_precursor_mass"] = np.nan
    return out


def merge_metadata_into_fragment_table(df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Merge optional MGF metadata into the fragment summary table."""
    if df.empty or metadata_df.empty:
        return df

    left = _add_merge_helpers(df)
    right = metadata_df.copy()
    right["metadata_scan_key"] = right["metadata_scan_key"].astype(str).str.strip()

    merged = left.merge(
        right,
        on=["batch_key", "metadata_scan_key"],
        how="left",
        suffixes=("", "_mgf"),
    )

    # Keep the original table batch name after merging on normalized batch_key.
    if "batch_x" in merged.columns and "batch" not in merged.columns:
        merged = merged.rename(columns={"batch_x": "batch"})
    if "batch_y" in merged.columns:
        merged = merged.drop(columns=["batch_y"], errors="ignore")

    # Fallback: if scan-based merge failed for some rows, fill by row order within each batch.
    missing = merged["metadata_order"].isna() if "metadata_order" in merged.columns else pd.Series(False, index=merged.index)
    if missing.any() and "batch" in df.columns:
        order_left = df.copy()
        order_left["batch_key"] = order_left["batch"].apply(_norm_batch)
        order_left["metadata_order"] = order_left.groupby("batch_key").cumcount() + 1
        order_merged = order_left.merge(
            right,
            on=["batch_key", "metadata_order"],
            how="left",
            suffixes=("", "_mgf"),
        )
        for col in right.columns:
            if col in ["batch", "metadata_order", "metadata_scan_key", "metadata_precursor_mass"]:
                continue
            if col in merged.columns and col in order_merged.columns:
                merged.loc[missing, col] = merged.loc[missing, col].fillna(order_merged.loc[missing, col])

    helper_cols = [
        "metadata_scan_key",
        "metadata_precursor_mass",
        "metadata_precursor_mass_mgf",
        "metadata_order",
        "batch_key",
        "batch_mgf",
    ]
    merged = merged.drop(columns=[c for c in helper_cols if c in merged.columns], errors="ignore")

    # Ensure these columns are visibly present even when the current MGF lacks them.
    for col in OPTIONAL_METADATA_FIELDS:
        if col not in merged.columns:
            merged[col] = ""

    preferred_cols = [
        "batch",
        "compound_name",
        "inchikey",
        "smiles",
        "inchi",
        "molecular_formula",
        "exact_mass",
        "adduct",
        "charge",
        "rt_seconds",
        "rt_minutes",
        "collision_energy",
        "ion_mode",
        "instrument",
        "scans",
        "scan_number",
        "precursor_mass",
        "n_fragments",
        "fragments",
    ]
    existing = [c for c in preferred_cols if c in merged.columns]
    rest = [c for c in merged.columns if c not in existing]
    return merged[existing + rest]


# -----------------------------
# General helpers
# -----------------------------
def _ensure_dir_with_mgfs_from_upload(files: List) -> str:
    """Write uploaded files into a temp directory and return its path."""
    tmpdir = tempfile.mkdtemp(prefix="mgf_uploads_")
    for f in files:
        name = Path(f.name).name
        out = Path(tmpdir) / name
        with open(out, "wb") as w:
            w.write(f.getbuffer())
    return tmpdir


def _ensure_dir_from_path_or_file(path_str: str) -> str:
    """
    Accept a directory OR a single .mgf file path.
    If it's a single .mgf file, copy it into a temp folder and return that folder.
    """
    p = Path(path_str).expanduser()
    if p.is_dir():
        return str(p)
    if p.is_file() and p.suffix.lower() == ".mgf":
        tmpdir = tempfile.mkdtemp(prefix="mgf_single_")
        shutil.copy2(str(p), str(Path(tmpdir) / p.name))
        return tmpdir
    raise FileNotFoundError(
        "Provide a folder containing .mgf files or a single .mgf file path."
    )


def _parse_frag_string(frag_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 'mz:rel%;mz:rel%;...' → (mz_array, rel_array)
    Returns empty arrays if string is blank.
    """
    if not frag_str:
        return np.array([]), np.array([])
    mzs, rels = [], []
    for part in frag_str.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        mz_txt, rel_txt = part.split(":", 1)
        try:
            mzs.append(float(mz_txt))
            rels.append(float(rel_txt.replace("%", "")))
        except ValueError:
            continue
    return np.array(mzs, float), np.array(rels, float)


@st.cache_data(show_spinner=False)
def build_table_from_dir(
    folder_with_mgfs: str,
    top_n: int,
    min_rel_pct: float,
    include_metadata: bool = True,
) -> pd.DataFrame:
    """Load spectra, build the summary DataFrame and append optional MGF metadata."""
    spectra_by_batch = mgf2frag.load_mgf_spectra(folder_with_mgfs)
    df = mgf2frag.spectra_to_dataframe(
        spectra_by_batch,
        top_n=top_n,
        min_rel_pct=min_rel_pct,
    )

    if include_metadata:
        metadata_df = extract_optional_metadata_from_dir(folder_with_mgfs)
        df = merge_metadata_into_fragment_table(df, metadata_df)

    cols = [
        "batch",
        "compound_name",
        "inchikey",
        "smiles",
        "inchi",
        "molecular_formula",
        "exact_mass",
        "adduct",
        "charge",
        "rt_seconds",
        "rt_minutes",
        "collision_energy",
        "ion_mode",
        "instrument",
        "scans",
        "scan_number",
        "precursor_mass",
        "n_fragments",
        "fragments",
    ]
    existing = [c for c in cols if c in df.columns]
    rest = [c for c in df.columns if c not in existing]
    return df[existing + rest]


# -----------------------------
# UI
# -----------------------------
st.title("MGF Survey → Fragment Summary Table")
st.caption(APP_VERSION)

with st.sidebar:
    st.header("Input options")
    mode = st.radio(
        "Choose how to provide MGF data:",
        ["Upload .mgf file(s)", "Use local path (folder or single .mgf)"],
    )

    st.header("Fragment parameters")
    top_n = st.number_input("Top N fragments to keep", 1, 50, 6, 1)
    min_rel_pct = st.number_input("Min relative intensity (%)", 0.0, 100.0, 1.0, 0.5)
    include_metadata = st.checkbox(
        "Read optional MGF metadata",
        value=True,
        help="Adds fields such as INCHIKEY, SMILES, FORMULA, NAME, ADDUCT, RT and CE when present in the MGF.",
    )
    st.caption("Fragments are filtered by relative intensity, then Top N retained.")

    st.divider()
    st.header("Search / Filter")
    q = st.text_input("Filter rows (simple substring over all columns)", "")
    st.caption("Tip: try a batch name, scan number, m/z value, InChIKey, SMILES or compound name.")

VIDEO_URL = "https://youtu.be/qeU8rRxtwXk"
try:
    st.sidebar.link_button("Video", VIDEO_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{VIDEO_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">📘 Tutorial</button>'
        "</a>",
        unsafe_allow_html=True,
    )

# Collect input
temp_dir_to_use = None
if mode == "Upload .mgf file(s)":
    uploads = st.file_uploader(
        "Drop one or more .mgf files here",
        type=["mgf"],
        accept_multiple_files=True,
    )
    go_btn = st.button("Build table from uploads", type="primary", disabled=not uploads)
    if go_btn and uploads:
        temp_dir_to_use = _ensure_dir_with_mgfs_from_upload(uploads)
else:
    folder_str = st.text_input("Folder path or single .mgf path", "")
    go_btn = st.button(
        "Build table from path", type="primary", disabled=(folder_str.strip() == "")
    )
    if go_btn and folder_str.strip():
        try:
            temp_dir_to_use = _ensure_dir_from_path_or_file(folder_str.strip())
        except Exception as e:
            st.error(str(e))
            temp_dir_to_use = None

# Build table
df = None
if temp_dir_to_use:
    with st.spinner("Reading MGF and assembling table..."):
        try:
            df = build_table_from_dir(
                temp_dir_to_use,
                int(top_n),
                float(min_rel_pct),
                bool(include_metadata),
            )
        except Exception as e:
            st.error(f"Failed to build table: {e}")

# Display and interactions
if df is not None and len(df):
    if q.strip():
        qlow = q.strip()
        mask = df.astype(str).apply(lambda c: c.str.contains(qlow, case=False, na=False)).any(axis=1)
        df_show = df[mask]
    else:
        df_show = df.copy()

    st.success(f"Loaded {len(df)} rows. Showing {len(df_show)} after filter.")

    if include_metadata:
        metadata_cols = [
            "inchikey",
            "smiles",
            "inchi",
            "molecular_formula",
            "compound_name",
            "adduct",
            "rt_seconds",
            "rt_minutes",
            "collision_energy",
        ]
        detected = [c for c in metadata_cols if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
        if detected:
            st.caption("Optional metadata detected: " + ", ".join(detected))
        else:
            st.caption("No optional annotation metadata detected in the uploaded MGF fields.")

        with st.expander("Metadata check", expanded=False):
            st.write("Columns in current table:", list(df.columns))
            preview_cols = [c for c in ["batch", "compound_name", "inchikey", "smiles", "molecular_formula", "adduct", "rt_seconds", "rt_minutes"] if c in df.columns]
            if preview_cols:
                st.dataframe(df[preview_cols].head(10), use_container_width=True)

    st.dataframe(df_show, use_container_width=True, height=520)

    st.subheader("Inspect fragments from a selected row")
    if "fragments" in df_show.columns:
        def label_for_row(idx, row):
            precursor = row.get("precursor_mass", "")
            scans = row.get("scans", row.get("scan_number", ""))
            batch = row.get("batch", "")
            name = row.get("compound_name", "")
            ik = row.get("inchikey", "")
            annotation = f" / {name}" if str(name).strip() else ""
            ik_txt = f" / InChIKey={ik}" if str(ik).strip() else ""
            return f"{idx} — {batch} / scans={scans} / precursor≈{precursor}{annotation}{ik_txt}"

        options = [(i, label_for_row(i, r)) for i, r in df_show.iterrows()]
        if options:
            label_to_idx = {lab: i for i, lab in options}
            selected_label = st.selectbox(
                "Pick a row to preview fragments", options=[lab for _, lab in options]
            )
            idx = label_to_idx[selected_label]
            row = df_show.loc[idx]

            frag_str = row.get("fragments", "")
            mzs, rels = _parse_frag_string(frag_str)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.write("**Fragment preview (relative intensity %)**")
                if mzs.size:
                    fig = go.Figure()
                    for mz_value, rel_value in zip(mzs, rels):
                        fig.add_trace(
                            go.Scatter(
                                x=[mz_value, mz_value],
                                y=[0, rel_value],
                                mode="lines",
                                hovertemplate="m/z=%{x}<br>%=%{y}<extra></extra>",
                                showlegend=False,
                            )
                        )
                    fig.update_layout(
                        xaxis_title="m/z",
                        yaxis_title="Relative intensity (%)",
                        template="simple_white",
                        height=360,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No fragments to display for this row.")

            with c2:
                st.markdown("**Raw fields**")
                raw_lines = [
                    f"batch: {row.get('batch','')}",
                    f"compound_name: {row.get('compound_name','')}",
                    f"inchikey: {row.get('inchikey','')}",
                    f"smiles: {row.get('smiles','')}",
                    f"molecular_formula: {row.get('molecular_formula','')}",
                    f"adduct: {row.get('adduct','')}",
                    f"rt_seconds: {row.get('rt_seconds','')}",
                    f"rt_minutes: {row.get('rt_minutes','')}",
                    f"collision_energy: {row.get('collision_energy','')}",
                    f"scans: {row.get('scans','')}",
                    f"scan_number: {row.get('scan_number','')}",
                    f"precursor_mass: {row.get('precursor_mass','')}",
                    f"n_fragments: {row.get('n_fragments','')}",
                    f"fragments: {frag_str}",
                ]
                st.code("\n".join(raw_lines))

    st.subheader("Export")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    tsv_bytes = df.to_csv(index=False, sep="\t").encode("utf-8")

    c_csv, c_tsv = st.columns(2)
    with c_csv:
        st.download_button(
            "Download table as CSV",
            data=csv_bytes,
            file_name="mgf_fragments_summary.csv",
            mime="text/csv",
        )
    with c_tsv:
        st.download_button(
            "Download table as TSV",
            data=tsv_bytes,
            file_name="mgf_fragments_summary.tsv",
            mime="text/tab-separated-values",
        )
elif df is not None and len(df) == 0:
    st.warning("No spectra were detected in the selected MGF data.")
else:
    st.info("Load your .mgf data (upload files or provide a path) and click **Build table**.")
