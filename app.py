# streamlit app for MGF → fragments table
# Save as: app.py
# Run: streamlit run app.py

# --- Robust imports for cloud -----------------------------------------------
import os
from pathlib import Path
import importlib.util
import streamlit as st
from typing import List, Tuple
import tempfile, shutil
from typing import List, Tuple
import tempfile, shutil

st.set_page_config(page_title="MGF → Fragment Tables", layout="wide")

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
    from pyteomics import mgf as _mgf_test  # <- ensures pyteomics is present
except ModuleNotFoundError as e:
    st.error(
        "A required package is missing: "
        f"**{e.name}**. On Streamlit Cloud, make sure it's listed in `requirements.txt` "
        "(see the app README)."
    )
    st.stop()

HERE = Path(__file__).resolve().parent
mgf2frag = None

# 1) normal import
try:
    import mgf_2_fragTable as mgf2frag  # noqa: E402
except ModuleNotFoundError:
    # 2) local side-load
    mgf2frag = _import_local_module("mgf_2_fragTable", HERE)

if mgf2frag is None:
    st.error("`mgf_2_fragTable.py` not found. Place it beside `app.py` in the repo.")
    st.stop()


# Your module (must be in the same folder as this app)
# Uses pyteomics under the hood
try:
    import mgf_2_fragTable as mgf2frag
except Exception as e:
    st.error(f"Could not import mgf_2_fragTable: {e}")
    st.stop()


# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir_with_mgfs_from_upload(files: List) -> str:
    """Write uploaded files into a temp directory and return its path."""
    tmpdir = tempfile.mkdtemp(prefix="mgf_uploads_")
    for f in files:
        # Some browsers may send full path; keep only name
        name = Path(f.name).name
        out = Path(tmpdir) / name
        with open(out, "wb") as w:
            w.write(f.read())
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
        if not part:
            continue
        if ":" not in part:
            continue
        mz_txt, rel_txt = part.split(":", 1)
        try:
            mzs.append(float(mz_txt))
            rels.append(float(rel_txt.replace("%", "")))
        except Exception:
            pass
    return np.array(mzs, float), np.array(rels, float)


@st.cache_data(show_spinner=False)
def build_table_from_dir(
    folder_with_mgfs: str,
    top_n: int,
    min_rel_pct: float,
) -> pd.DataFrame:
    """Load spectra and build the summary DataFrame."""
    spectra_by_batch = mgf2frag.load_mgf_spectra(folder_with_mgfs)
    df = mgf2frag.spectra_to_dataframe(
        spectra_by_batch,
        top_n=top_n,
        min_rel_pct=min_rel_pct,
    )
    # Safer ordering of columns if present
    cols = [
        "batch",
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
st.set_page_config(page_title="MGF → Fragment Table", layout="wide")
st.title("MGF Survey → Fragment Summary Table")

with st.sidebar:
    st.header("Input options")
    mode = st.radio(
        "Choose how to provide MGF data:",
        ["Upload .mgf file(s)", "Use local path (folder or single .mgf)"],
    )
    top_n = st.number_input("Top N fragments to keep", 1, 50, 6, 1)
    min_rel_pct = st.number_input("Min relative intensity (%)", 0.0, 100.0, 1.0, 0.5)
    st.caption("Fragments are filtered by relative intensity, then Top N retained.")

    st.divider()
    st.header("Search / Filter")
    q = st.text_input("Filter rows (simple substring over all columns)", "")
    st.caption("Tip: try a batch name, scan number, or m/z value.")

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
    go_btn = st.button("Build table from path", type="primary", disabled=(folder_str.strip() == ""))
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
            df = build_table_from_dir(temp_dir_to_use, int(top_n), float(min_rel_pct))
        except Exception as e:
            st.error(f"Failed to build table: {e}")

# Display and interactions
if df is not None and len(df):
    # Optional filter
    if q.strip():
        qlow = q.strip().lower()
        df_show = df[df.apply(lambda r: qlow in (" ".join(map(str, r.values))).lower(), axis=1)]
    else:
        df_show = df.copy()

    st.success(f"Loaded {len(df)} rows. Showing {len(df_show)} after filter.")
    st.dataframe(df_show, use_container_width=True, height=520)

    # Row selection by index
    st.subheader("Inspect fragments from a selected row")
    if "fragments" in df_show.columns:
        # Build choices as "index — batch / scan / precursor"
        def label_for_row(idx, row):
            precursor = row.get("precursor_mass", "")
            scans = row.get("scans", "")
            batch = row.get("batch", "")
            return f"{idx} — {batch} / scans={scans} / precursor≈{precursor}"

        options = [(i, label_for_row(i, r)) for i, r in df_show.iterrows()]
        if options:
            selected_label = st.selectbox(
                "Pick a row to preview fragments", options=[lab for _, lab in options]
            )
            # Find the original index from label
            idx = dict(options)[selected_label]
            row = df_show.loc[idx]

            frag_str = row.get("fragments", "")
            mzs, rels = _parse_frag_string(frag_str)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.write("**Fragment preview (relative intensity %)**")
                if mzs.size:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=mzs,
                            y=rels,
                            hovertemplate="m/z=%{x}<br>%=%{y}",
                        )
                    )
                    fig.update_layout(
                        xaxis_title="m/z",
                        yaxis_title="Relative intensity (%)",
                        bargap=0.1,
                        template="simple_white",
                        height=360,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No fragments to display for this row.")

            with c2:
                st.markdown("**Raw fields**")
                st.code(
                    f"batch: {row.get('batch','')}\n"
                    f"scans: {row.get('scans','')}\n"
                    f"scan_number: {row.get('scan_number','')}\n"
                    f"precursor_mass: {row.get('precursor_mass','')}\n"
                    f"n_fragments: {row.get('n_fragments','')}\n"
                    f"fragments: {frag_str}"
                )

    # Download CSV
    st.subheader("Export")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table as CSV",
        data=csv_bytes,
        file_name="mgf_fragments_summary.csv",
        mime="text/csv",
    )
else:
    st.info("Load your .mgf data (upload files or provide a path) and click **Build table**.")
