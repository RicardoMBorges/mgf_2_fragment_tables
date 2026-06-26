# streamlit app for MGF → fragments table
# Save as: app.py
# Run: streamlit run app.py

from pathlib import Path
import tempfile
import shutil
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

APP_VERSION = "v0.5 — MassQL-ready fragment m/z column"

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

try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
except ModuleNotFoundError as e:
    st.error(
        "A required package is missing: "
        f"**{e.name}**. On Streamlit Cloud, make sure it is listed in `requirements.txt`."
    )
    st.stop()

try:
    from PIL import Image

    STATIC_DIR = Path(__file__).parent / "static"
    LOGO_PATH = STATIC_DIR / "LAABio.png"
    logo = Image.open(LOGO_PATH)
    st.sidebar.image(logo, use_container_width=True)
except Exception:
    st.sidebar.warning("Logo not found at static/LAABio.png")


# -----------------------------
# MGF metadata definitions
# -----------------------------
OPTIONAL_METADATA_FIELDS: Dict[str, List[str]] = {
    "compound_name": [
        "name",
        "compound_name",
        "compoundname",
        "compound",
        "title",
        "synonym",
        "spectrum_name",
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
    "library_id": [
        "spectrumid",
        "spectrum_id",
        "library_id",
        "libraryid",
        "spectrum_id_in_library",
    ],
}


def _norm_key(key: Any) -> str:
    """Normalize metadata keys so INCHIKEY, INCHI_KEY and InChI-Key all match."""
    return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_available_param(params: Dict[str, Any], aliases: List[str]) -> str:
    normalized = {_norm_key(k): v for k, v in params.items()}
    for alias in aliases:
        value = _clean_value(normalized.get(_norm_key(alias), ""))
        if value:
            return value
    return ""


def _parse_first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        txt = str(value).replace(",", " ").split()[0]
        return float(txt)
    except Exception:
        return None


def _format_float(value: float, ndigits: int = 6) -> str:
    txt = f"{value:.{ndigits}f}"
    return txt.rstrip("0").rstrip(".")


def _parse_peak_line(line: str) -> Optional[Tuple[float, float]]:
    """
    Parse MGF peak lines like:
    123.0456 98765
    123.0456 98765 "annotation"
    """
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        mz = float(parts[0])
        intensity = float(parts[1])
        return mz, intensity
    except ValueError:
        return None


def _parse_mgf_file_direct(mgf_path: Path) -> List[Dict[str, Any]]:
    """
    Parse one MGF file directly.

    This avoids relying on a second metadata merge step. Metadata and fragments
    are extracted from the same BEGIN IONS block, so optional fields such as
    INCHIKEY, SMILES, FORMULA, RT and CE stay attached to the correct spectrum.
    """
    spectra: List[Dict[str, Any]] = []
    in_block = False
    params: Dict[str, str] = {}
    peaks: List[Tuple[float, float]] = []

    with open(mgf_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()

            if upper == "BEGIN IONS":
                in_block = True
                params = {}
                peaks = []
                continue

            if upper == "END IONS" and in_block:
                spectra.append({"params": params, "peaks": peaks})
                in_block = False
                params = {}
                peaks = []
                continue

            if not in_block:
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Keep original key but handle repeated keys safely.
                if key in params and params[key] != value:
                    params[key] = f"{params[key]};{value}"
                else:
                    params[key] = value
            else:
                parsed_peak = _parse_peak_line(line)
                if parsed_peak is not None:
                    peaks.append(parsed_peak)

    return spectra


def _spectrum_to_record(
    batch_name: str,
    spectrum_index: int,
    params: Dict[str, Any],
    peaks: List[Tuple[float, float]],
    top_n: int,
    min_rel_pct: float,
    include_metadata: bool,
) -> Dict[str, Any]:
    precursor_mass = _parse_first_float(
        _first_available_param(params, ["pepmass", "precursor_mass", "precursormz", "precursor_mz"])
    )

    scans = _first_available_param(params, ["scans", "scan", "scan_number", "spectrumid", "spectrum_id"])
    scan_number = scans if scans else str(spectrum_index)

    record: Dict[str, Any] = {
        "batch": batch_name,
        "scans": scans,
        "scan_number": scan_number,
        "precursor_mass": precursor_mass if precursor_mass is not None else "",
    }

    if include_metadata:
        for out_col, aliases in OPTIONAL_METADATA_FIELDS.items():
            record[out_col] = _first_available_param(params, aliases)

    if peaks:
        mzs = np.array([p[0] for p in peaks], dtype=float)
        intensities = np.array([p[1] for p in peaks], dtype=float)

        max_intensity = float(np.nanmax(intensities)) if intensities.size else 0.0
        if max_intensity > 0:
            rels = intensities / max_intensity * 100.0
        else:
            rels = np.zeros_like(intensities)

        mask = rels >= float(min_rel_pct)
        filtered = list(zip(mzs[mask], rels[mask], intensities[mask]))

        # Keep top N by relative intensity, then sort by m/z for a clean spectrum string.
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[: int(top_n)]
        filtered = sorted(filtered, key=lambda x: x[0])

        fragments_mz = ";".join(_format_float(mz) for mz, _rel, _int in filtered)
        fragments_with_intensity = ";".join(
            f"{_format_float(mz)}:{_format_float(rel, 2)}%" for mz, rel, _int in filtered
        )

        record["n_fragments_original"] = len(peaks)
        record["n_fragments"] = len(filtered)
        # MassQL Builder-ready column: m/z values only, separated by semicolon.
        record["fragments"] = fragments_mz
        # Human-readable spectrum summary retaining relative intensities.
        record["fragments_with_intensity"] = fragments_with_intensity
    else:
        record["n_fragments_original"] = 0
        record["n_fragments"] = 0
        record["fragments"] = ""
        record["fragments_with_intensity"] = ""

    return record


@st.cache_data(show_spinner=False)
def build_table_from_dir(
    folder_with_mgfs: str,
    top_n: int,
    min_rel_pct: float,
    include_metadata: bool = True,
) -> pd.DataFrame:
    """Build the fragment table directly from MGF blocks."""
    folder = Path(folder_with_mgfs)
    records: List[Dict[str, Any]] = []

    for mgf_path in sorted(folder.glob("*.mgf")):
        spectra = _parse_mgf_file_direct(mgf_path)
        for i, spectrum in enumerate(spectra, start=1):
            records.append(
                _spectrum_to_record(
                    batch_name=mgf_path.name,
                    spectrum_index=i,
                    params=spectrum["params"],
                    peaks=spectrum["peaks"],
                    top_n=int(top_n),
                    min_rel_pct=float(min_rel_pct),
                    include_metadata=bool(include_metadata),
                )
            )

    df = pd.DataFrame(records)

    # Ensure optional metadata columns are always visible and exported.
    if include_metadata:
        for col in OPTIONAL_METADATA_FIELDS:
            if col not in df.columns:
                df[col] = ""

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
        "library_id",
        "scans",
        "scan_number",
        "precursor_mass",
        "n_fragments_original",
        "n_fragments",
        "fragments",
        "fragments_with_intensity",
    ]
    existing = [c for c in preferred_cols if c in df.columns]
    rest = [c for c in df.columns if c not in existing]
    return df[existing + rest]


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
    """Accept a directory OR a single .mgf file path."""
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
    """Convert 'mz:rel%;mz:rel%;...' → (mz_array, rel_array)."""
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

if df is not None and len(df):
    if q.strip():
        qlow = q.strip()
        mask = df.astype(str).apply(
            lambda c: c.str.contains(qlow, case=False, na=False, regex=False)
        ).any(axis=1)
        df_show = df[mask]
    else:
        df_show = df.copy()

    st.success(f"Loaded {len(df)} spectra. Showing {len(df_show)} after filter.")

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
            "ion_mode",
            "instrument",
            "library_id",
        ]
        detected = [
            c for c in metadata_cols
            if c in df.columns and df[c].astype(str).str.strip().ne("").any()
        ]
        if detected:
            st.caption("Optional metadata detected: " + ", ".join(detected))
        else:
            st.caption("No optional annotation metadata detected in the uploaded MGF fields.")

        with st.expander("Metadata check", expanded=False):
            preview_cols = [
                c for c in [
                    "batch",
                    "compound_name",
                    "inchikey",
                    "smiles",
                    "molecular_formula",
                    "adduct",
                    "rt_seconds",
                    "rt_minutes",
                    "collision_energy",
                ]
                if c in df.columns
            ]
            st.write("Columns in current table:", list(df.columns))
            st.dataframe(df[preview_cols].head(20), use_container_width=True)

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

            frag_str = row.get("fragments_with_intensity", row.get("fragments", ""))
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
                    f"n_fragments_original: {row.get('n_fragments_original','')}",
                    f"n_fragments: {row.get('n_fragments','')}",
                    f"fragments: {row.get('fragments','')}",
                    f"fragments_with_intensity: {row.get('fragments_with_intensity','')}",
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
