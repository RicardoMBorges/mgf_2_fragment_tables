# streamlit app for MGF → fragments table
# Save as: app.py
# Run: streamlit run app.py

# app.py — safe bootstrap
import os, sys, importlib.util
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="MGF → Fragment Tables", layout="wide")

st.title("MGF → Fragment Tables")
st.caption("Bootstrap check — if you see this, the app is rendering correctly.")

with st.expander("Environment & versions"):
    st.write({"python": sys.version, "cwd": os.getcwd(), "files": os.listdir()})

# Try to import your helper and reveal any error visibly
def _import_local(mod_name: str, base: Path = Path(".")):
    p = base / f"{mod_name}.py"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    spec = importlib.util.spec_from_file_location(mod_name, p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        st.error(f"Import error in `{mod_name}.py`")
        st.exception(e)
        raise
    return mod

try:
    mgf2frag = _import_local("mgf_2_fragTable")
    st.success("`mgf_2_fragTable.py` imported successfully.")
except Exception:
    st.stop()

st.markdown("### Next step")
st.write("Now re-enable your full UI piece by piece below this line.")
